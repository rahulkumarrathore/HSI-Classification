import numpy as np
import scipy.io as sio
from scipy.ndimage import maximum_filter
from sklearn.decomposition import PCA

from Dataste_directory import PU_CUBE_PATH, PU_GT_PATH

# ============================================================
# FLAGS (hardcoded -- yahin badlo)
# ============================================================

PATCH = 13               # SSFTT patch size; buffer + padding = PATCH // 2
TEST_RATIO = 0.30
VAL_RATIO = 0.10         # baaki = train
SEED = 42
PCA_K = 30               # SSFTT ko 30 bands chahiye

CLASS_NAMES = [
    "Asphalt", "Meadows", "Gravel", "Trees", "Painted-metal-sheets",
    "Bare-Soil", "Bitumen", "Self-Blocking-Bricks", "Shadows",
]

MARGIN = PATCH // 2


# ============================================================
# 1. LOAD
# ============================================================

def load_pu():
    cube = sio.loadmat(PU_CUBE_PATH)["paviaU"].astype(np.float32)   # (610, 340, 103)
    gt = sio.loadmat(PU_GT_PATH)["paviaU_gt"].astype(np.int64)      # (610, 340), 0 = unlabelled
    return cube, gt


# ============================================================
# 2. SPLIT (on the fly, same SEED -> same split)
# ============================================================

def make_split(gt):
    """Per class ek cut lambe axis ke along: test | train | val.
    Test ke around MARGIN ka buffer -> wahan ke train/val pixels drop.
    Returns rows, cols, labels (0..8) aur {"train","val","test"} -> index arrays."""
    H, W = gt.shape
    rows, cols = np.nonzero(gt)
    labels = gt[rows, cols] - 1

    split = np.full(len(rows), "", dtype=object)
    for c in range(len(CLASS_NAMES)):
        idx = np.where(labels == c)[0]
        n = len(idx)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))

        r, cc = rows[idx], cols[idx]
        axis_vals = r if (r.max() - r.min()) >= (cc.max() - cc.min()) else cc
        order = idx[np.argsort(axis_vals, kind="stable")]

        split[order[:n_test]] = "test"
        split[order[n_test:n - n_val]] = "train"
        split[order[n - n_val:]] = "val"

    # buffer around test
    test_map = np.zeros((H, W), dtype=bool)
    test_map[rows[split == "test"], cols[split == "test"]] = True
    near_test = maximum_filter(test_map, size=2 * MARGIN + 1)
    split[(split != "test") & near_test[rows, cols]] = "buffer"

    # train/val ko SEED se shuffle (training order ke liye); spatial position same rehti hai
    rng = np.random.default_rng(SEED)
    idx = {s: rng.permutation(np.where(split == s)[0]) for s in ["train", "val", "test"]}
    return rows, cols, labels, idx


def leakage_percent(gt, rows, cols, idx):
    """Kitne % test pixels kisi train patch ke andar aate hain (0 hona chahiye)."""
    train_map = np.zeros(gt.shape, dtype=bool)
    train_map[rows[idx["train"]], cols[idx["train"]]] = True
    in_train_patch = maximum_filter(train_map, size=2 * MARGIN + 1)
    t = idx["test"]
    return 100.0 * in_train_patch[rows[t], cols[t]].mean()


# ============================================================
# 3 + 4. NORMALISE + PCA (sirf train pixels par fit)
# ============================================================

def normalise_and_pca(cube, rows, cols, train_idx):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    train_px = cube[rows[train_idx], cols[train_idx]]               # (n_train, 103)

    mean = train_px.mean(axis=0)
    std = train_px.std(axis=0) + 1e-8
    flat = (flat - mean) / std
    train_px = (train_px - mean) / std

    pca = PCA(n_components=PCA_K, whiten=True, random_state=SEED)
    pca.fit(train_px)
    reduced = pca.transform(flat).astype(np.float32).reshape(H, W, PCA_K)
    return reduced, pca


# ============================================================
# 5. PADDING
# ============================================================

def pad_cube(cube):
    return np.pad(cube, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode="reflect")


# ============================================================
# FULL PIPELINE (train.py / test.py yahi call karenge)
# ============================================================

def preprocess():
    cube, gt = load_pu()
    rows, cols, labels, idx = make_split(gt)
    reduced, pca = normalise_and_pca(cube, rows, cols, idx["train"])
    cube_pad = pad_cube(reduced)
    return cube_pad, rows, cols, labels, idx


# ============================================================
# 6. CHECKS (sirf jab ye file directly chalao)
# ============================================================

if __name__ == "__main__":
    cube, gt = load_pu()
    print("cube", cube.shape, "gt", gt.shape)

    rows, cols, labels, idx = make_split(gt)
    print("labelled pixels", len(rows))
    print(f"{'class':<22}{'train':>7}{'val':>7}{'test':>7}")
    for c, name in enumerate(CLASS_NAMES):
        n = [np.sum(labels[idx[s]] == c) for s in ["train", "val", "test"]]
        print(f"{name:<22}{n[0]:>7}{n[1]:>7}{n[2]:>7}")
    print(f"{'TOTAL':<22}{len(idx['train']):>7}{len(idx['val']):>7}{len(idx['test']):>7}")
    print("buffer (dropped)", len(rows) - sum(len(v) for v in idx.values()))
    print(f"test leakage {leakage_percent(gt, rows, cols, idx):.2f}%")

    reduced, pca = normalise_and_pca(cube, rows, cols, idx["train"])
    print("after pca", reduced.shape,
          f"explained variance {100 * pca.explained_variance_ratio_.sum():.2f}%")

    cube_pad = pad_cube(reduced)
    print("after padding", cube_pad.shape)
