import os
import glob
import math
import json
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, radon

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib



# Utils: listar ficheiros

def list_image_files(folder, exts=("png", "jpg", "jpeg", "bmp", "tif", "tiff")):
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"*.{e}")))
        files.extend(glob.glob(os.path.join(folder, f"*.{e.upper()}")))
    return sorted(files)



# 1) Imagem 2D: carregar + preprocess

def read_and_preprocess_image(path, img_size=(128, 128)):
    img = imread(path)

    if img.ndim == 3:
        img = rgb2gray(img)

    img = img.astype(np.float32)
    img = resize(img, img_size, anti_aliasing=True).astype(np.float32)

    # Min-Max normalization
    img -= img.min()
    denom = img.max() - img.min()
    if denom > 0:
        img /= denom

    return img



# 2) Radon -> sinais 1D (canais = ângulos)
#    output por imagem: (n_angles, proj_len)

def radon_signals(img2d, n_angles=60, circle=False):
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    R = radon(img2d, theta=theta, circle=circle)
    R = R.T.astype(np.float32)

    # normalização por canal
    R -= R.mean(axis=1, keepdims=True)
    R /= (R.std(axis=1, keepdims=True) + 1e-8)

    return R



# Cache Radon em memmap (para datasets grandes)

def build_or_load_radon_cache(
    non_target_dir,
    target_dir,
    cache_dir="./cache_radon",
    img_size=(128, 128),
    n_angles=36,
    circle=True,
    limit_per_class=None  
):
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")
    data_path = os.path.join(cache_dir, "radon_memmap.dat")
    labels_path = os.path.join(cache_dir, "labels.npy")
    paths_path = os.path.join(cache_dir, "paths.npy")

    
    non_files = list_image_files(non_target_dir)
    tar_files = list_image_files(target_dir)

    if limit_per_class is not None:
        non_files = non_files[:limit_per_class]
        tar_files = tar_files[:limit_per_class]

    if len(non_files) == 0 or len(tar_files) == 0:
        raise FileNotFoundError("Não encontrei imagens nas pastas non_target/target.")

    all_files = non_files + tar_files
    y = np.array([0]*len(non_files) + [1]*len(tar_files), dtype=np.int64)

    
    if os.path.exists(meta_path) and os.path.exists(data_path) and os.path.exists(labels_path) and os.path.exists(paths_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        same = (
            meta.get("img_size") == list(img_size) and
            meta.get("n_angles") == n_angles and
            meta.get("circle") == circle and
            meta.get("n_samples") == len(all_files)
        )
        if same:
            labels = np.load(labels_path)
            paths = np.load(paths_path, allow_pickle=True)
            # abrir memmap
            shape = tuple(meta["shape"])  
            X_mm = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
            return X_mm, labels, paths

    
    print(f"[cache] A construir Radon cache para {len(all_files)} imagens...")
    
    test_img = read_and_preprocess_image(all_files[0], img_size=img_size)
    test_R = radon_signals(test_img, n_angles=n_angles, circle=circle)
    proj_len = test_R.shape[1]

    shape = (len(all_files), n_angles, proj_len)
    X_mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=shape)

    
    for i, p in enumerate(all_files):
        if (i+1) % 100 == 0:
            print(f"[cache] {i+1}/{len(all_files)}")
        img = read_and_preprocess_image(p, img_size=img_size)
        R = radon_signals(img, n_angles=n_angles, circle=circle)
        X_mm[i, :, :] = R

    X_mm.flush()
    np.save(labels_path, y)
    np.save(paths_path, np.array(all_files, dtype=object))

    meta = {
        "img_size": list(img_size),
        "n_angles": n_angles,
        "circle": circle,
        "n_samples": len(all_files),
        "shape": list(shape),
        "dtype": "float32",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    
    X_mm = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    return X_mm, y, np.array(all_files, dtype=object)



# 3) CSP (binário)

class CSP:
    def __init__(self, n_components=6, reg=1e-6):
        if n_components % 2 != 0:
            raise ValueError("n_components deve ser par (ex.: 4,6,8).")
        self.n_components = n_components
        self.reg = reg
        self.W_ = None

    @staticmethod
    def _cov(trial):
        
        C = trial @ trial.T
        C = C / (np.trace(C) + 1e-12)
        return C

    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Precisas de exemplos das 2 classes.")

        C0 = np.mean([self._cov(t) for t in X0], axis=0)
        C1 = np.mean([self._cov(t) for t in X1], axis=0)

        n_ch = C0.shape[0]
        C0 = C0 + self.reg * np.eye(n_ch, dtype=np.float64)
        C1 = C1 + self.reg * np.eye(n_ch, dtype=np.float64)

        Cc = C0 + C1

        
        eigvals, E = np.linalg.eigh(Cc)
        eigvals = np.maximum(eigvals, 1e-12)
        P = (E @ np.diag(1.0 / np.sqrt(eigvals)) @ E.T)

        S0 = P @ C0 @ P.T
        eigvals_s0, B = np.linalg.eigh(S0)
        idx = np.argsort(eigvals_s0)[::-1]
        B = B[:, idx]

        W = B.T @ P  
        m = self.n_components
        half = m // 2
        pick = list(range(half)) + list(range(n_ch - half, n_ch))
        self.W_ = W[pick, :].astype(np.float32)
        return self

    def transform(self, X):
        if self.W_ is None:
            raise RuntimeError("CSP não ajustado (fit primeiro).")
        feats = np.empty((len(X), self.W_.shape[0]), dtype=np.float32)
        for i, trial in enumerate(X):
            Z = self.W_ @ trial  
            var = np.var(Z, axis=1) + 1e-12
            feats[i] = np.log(var / np.sum(var))
        return feats

class RadonInputNormalizer:
    """
    Normaliza os sinais obtidos pela Transformada de Radon.

    A média e o desvio-padrão são calculados apenas no conjunto de treino,
    evitando data leakage.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# 4) Treino SVM (com split e CV)

def train_csp_svm(X, y, n_csp_components=10, kernel="rbf", C=50, gamma="scale", test_size=0.2, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # CSP
    csp = CSP(n_components=n_csp_components)
    csp.fit(X_tr, y_tr)

    F_tr = csp.transform(X_tr)
    F_te = csp.transform(X_te)

    # SVM com normalização dos inputs do classificador
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma))
    ])

    clf.fit(F_tr, y_tr)
    pred = clf.predict(F_te)

    print("\n=== Teste ===")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["non-target", "target"]))

    F_all = csp.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, F_all, y, cv=cv, scoring="f1")
    print(f"F1 (CV 5-fold, aproximado): mean={scores.mean():.3f} std={scores.std():.3f}")

    return csp, clf


def main():

    non_target_dir = "./data/non_target"
    target_dir = "./data/target"

    img_size = (128, 128)
    n_angles = 60
    circle = False

    X_mm, y, paths = build_or_load_radon_cache(
        non_target_dir=non_target_dir,
        target_dir=target_dir,
        cache_dir="./cache_radon",
        img_size=img_size,
        n_angles=n_angles,
        circle=circle,
        limit_per_class=None
    )

    X = np.asarray(X_mm)
    print(f"[info] X shape = {X.shape} (N, angles, proj_len) | y={y.shape}")

    csp, svm = train_csp_svm(
    X, y,
    n_csp_components=12,
    kernel="rbf",
    C=50,
    gamma="scale",
    test_size=0.2,
    seed=42
)

    os.makedirs("./models", exist_ok=True)
    joblib.dump(csp, "./models/csp.pkl")
    joblib.dump(svm, "./models/svm.pkl")

    print("\nModelos guardados em ./models/csp.pkl e ./models/svm.pkl")



if __name__ == "__main__":
    main()