import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Preprocessing import preprocess, PATCH, CLASS_NAMES

# ============================================================
# FLAGS (hardcoded -- yahin badlo)
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42


# ============================================================
# DATASET: pixel number -> (1, 30, 13, 13) patch + label
# ============================================================

class PUPatchDataset(Dataset):
    def __init__(self, cube_pad, rows, cols, labels, idx):
        self.cube_pad = cube_pad      # (622, 352, 30), PATCH//2 padding ke saath
        self.rows = rows
        self.cols = cols
        self.labels = labels
        self.idx = idx                # is split ke pixel numbers

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        r, c = self.rows[i], self.cols[i]
        # padding ki wajah se original (r, c) padded cube me (r+M, c+M) par hai,
        # to centred patch = cube_pad[r : r+PATCH, c : c+PATCH]
        patch = self.cube_pad[r:r + PATCH, c:c + PATCH, :]          # (13, 13, 30)
        patch = np.transpose(patch, (2, 0, 1))[None]                # (1, 30, 13, 13)
        x = torch.from_numpy(np.ascontiguousarray(patch)).float()
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


# ============================================================
# LOADERS
# ============================================================

def get_loaders():
    cube_pad, rows, cols, labels, idx = preprocess()

    datasets = {s: PUPatchDataset(cube_pad, rows, cols, labels, idx[s])
                for s in ["train", "val", "test"]}

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, generator=g)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader


# ============================================================
# CHECKS (sirf jab ye file directly chalao)
# ============================================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"{name:<6} samples {len(loader.dataset):>6}  batches {len(loader):>4}")

    x, y = next(iter(train_loader))
    print("batch x", tuple(x.shape), x.dtype)
    print("batch y", tuple(y.shape), y.dtype, "labels", y[:10].tolist())
    print("classes in batch", sorted({CLASS_NAMES[l] for l in y.tolist()}))

    # centre check: patch ka beech wala pixel == us pixel ka PCA spectrum
    ds = train_loader.dataset
    x0, _ = ds[0]
    i = ds.idx[0]
    m = PATCH // 2
    centre = x0[0, :, m, m].numpy()
    direct = ds.cube_pad[ds.rows[i] + m, ds.cols[i] + m, :]
    print("centre pixel match", np.allclose(centre, direct))
