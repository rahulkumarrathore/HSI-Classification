import os
import time
import random
import numpy as np
import torch
from torch import nn

from Dataloader import get_loaders
from Models import get_model
from Compute_metrics import compute_metrics, print_metrics

# ============================================================
# FLAGS (hardcoded -- yahin badlo)
# ============================================================

MODEL = "SSFTT"
EPOCHS = 26
LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = f"/home/rahul-kumar-rathore/data/HSI/OurHSI/runs/{MODEL}_spatial_p13"
BEST_PATH = os.path.join(SAVE_DIR, "best.pth")
LOG_PATH = os.path.join(SAVE_DIR, "train_log.txt")


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(DEVICE)).argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("model", MODEL, "| device", DEVICE, "| save", SAVE_DIR)

    train_loader, val_loader, _ = get_loaders()     # test loader yahan use NAHI hota -> Test.py
    model = get_model(MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_oa, best_epoch = -1.0, -1
    log = open(LOG_PATH, "w")
    log.write("epoch\tloss\tval_OA\tval_AA\tval_Kappa\ttime_s\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        y_true, y_pred = predict(model, val_loader)
        m = compute_metrics(y_true, y_pred)
        dt = time.time() - t0

        tag = ""
        if m["OA"] > best_oa:
            best_oa, best_epoch = m["OA"], epoch
            torch.save({"model": MODEL, "epoch": epoch, "state_dict": model.state_dict(),
                        "val_OA": float(m["OA"]), "val_AA": float(m["AA"]),
                        "val_Kappa": float(m["Kappa"])}, BEST_PATH)   # float(): numpy scalar torch.load me fail hota hai
            tag = "  <- best (saved)"

        print(f"epoch {epoch:3d}/{EPOCHS}  loss {loss:.4f}  "
              f"val OA {m['OA']:.2f}  AA {m['AA']:.2f}  Kappa {m['Kappa']:.2f}  ({dt:.1f}s){tag}")
        log.write(f"{epoch}\t{loss:.4f}\t{m['OA']:.2f}\t{m['AA']:.2f}\t{m['Kappa']:.2f}\t{dt:.1f}\n")
        log.flush()

    log.close()

    # best model ka val result
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    y_true, y_pred = predict(model, val_loader)
    print_metrics(compute_metrics(y_true, y_pred), f"VAL (best epoch {best_epoch})")
    print("saved", BEST_PATH)
    print("log  ", LOG_PATH)
