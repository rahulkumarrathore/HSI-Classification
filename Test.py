import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from Dataloader import get_loaders
from Models import get_model
from Compute_metrics import compute_metrics, print_metrics, CLASS_NAMES

# ============================================================
# FLAGS (hardcoded -- yahin badlo; Train.py wale MODEL/SAVE_DIR se match karo)
# ============================================================

MODEL = "SSFTT"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = f"/home/rahul-kumar-rathore/data/HSI/OurHSI/runs/{MODEL}_spatial_p13"
BEST_PATH = os.path.join(SAVE_DIR, "best.pth")
RESULT_PATH = os.path.join(SAVE_DIR, "test_result.txt")
MAP_PATH = os.path.join(SAVE_DIR, "test_map.png")

H, W = 610, 340          # PaviaU image size
COLORS = ["#c0c0c0", "#00ff00", "#00ffff", "#008000", "#ff00ff",
          "#a0522d", "#800080", "#ff0000", "#ffff00"]          # 9 classes


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(DEVICE)).argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def save_map(rows, cols, y_true, y_pred, path):
    """Sirf test pixels: GT vs prediction vs galat pixels."""
    gt_map = np.full((H, W), -1)
    pr_map = np.full((H, W), -1)
    gt_map[rows, cols] = y_true
    pr_map[rows, cols] = y_pred
    wrong = np.zeros((H, W))
    wrong[rows, cols] = (y_true != y_pred)

    cmap = ListedColormap(["black"] + COLORS)          # -1 (no test pixel) = black
    fig, ax = plt.subplots(1, 3, figsize=(12, 7))
    ax[0].imshow(gt_map + 1, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax[0].set_title("Test GT")
    ax[1].imshow(pr_map + 1, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax[1].set_title(f"{MODEL} prediction")
    ax[2].imshow(wrong, cmap="gray", interpolation="nearest")
    ax[2].set_title("Wrong (white)")
    for a in ax:
        a.axis("off")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)],
               loc="lower center", ncol=5, fontsize=8)
    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("model", MODEL, "| device", DEVICE, "| ckpt", BEST_PATH)

    _, _, test_loader = get_loaders()              # test loader shuffle=False -> order == idx order
    model = get_model(MODEL).to(DEVICE)
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded best epoch {ckpt['epoch']}  (val OA {ckpt['val_OA']:.2f})")

    y_true, y_pred = predict(model, test_loader)
    m = compute_metrics(y_true, y_pred)
    print_metrics(m, f"TEST {MODEL}")

    # result text file
    with open(RESULT_PATH, "w") as f:
        f.write(f"model {MODEL}  best epoch {ckpt['epoch']}  val OA {ckpt['val_OA']:.2f}\n")
        f.write(f"TEST OA {m['OA']:.2f}  AA {m['AA']:.2f}  Kappa {m['Kappa']:.2f}\n\n")
        for name, acc, n in zip(CLASS_NAMES, m["per_class"], m["cm"].sum(axis=1)):
            f.write(f"{name:<22}{acc:7.2f}  (n={n})\n")
        f.write("\nconfusion matrix (rows = true, cols = pred)\n")
        f.write(np.array2string(m["cm"], max_line_width=200) + "\n")

    # classification map (sirf test pixels)
    ds = test_loader.dataset
    save_map(ds.rows[ds.idx], ds.cols[ds.idx], y_true, y_pred, MAP_PATH)

    print("saved", RESULT_PATH)
    print("saved", MAP_PATH)
