import os
import importlib.util
import torch
from torch import nn

# ============================================================
# FLAGS (hardcoded -- yahin badlo)
# ============================================================

SSFTT_DIR = os.path.dirname(os.path.abspath(__file__))   # SSFTTnet.py isi folder me (original: SAMnetHSI/HSI_SSFTT/cls_SSFTT_IP)
NUM_CLASSES = 9          # PaviaU
BANDS = 30               # PCA_K (SSFTT me 8*28 hardcoded hai -> 30 hi chahiye)
PATCH = 13

# ============================================================
# SSFTT (original code as-is import)
# ============================================================

# file path se seedha load -- sys.path me pura SSFTT folder nahi daalte
# (wahan IP_train.py, get_cls_map.py bhi hain), aur IDE "import not resolved" nahi dikhata
_spec = importlib.util.spec_from_file_location("SSFTTnet", os.path.join(SSFTT_DIR, "SSFTTnet.py"))
_ssftt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ssftt)
SSFTTnet = _ssftt.SSFTTnet


# ============================================================
# MODEL CHOOSER (Train.py / Test.py yahi call karenge)
# ============================================================

def get_model(name):
    if name == "SSFTT":
        return SSFTTnet(num_classes=NUM_CLASSES)
    raise ValueError(f"unknown model {name}")


# ============================================================
# CHECKS (sirf jab ye file directly chalao)
# ============================================================

if __name__ == "__main__":
    x = torch.randn(4, 1, BANDS, PATCH, PATCH)
    for name in ["SSFTT"]:
        model = get_model(name).eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            y = model(x)
        print(f"{name:<9} in {tuple(x.shape)} -> out {tuple(y.shape)}   params {n_params:,}")
