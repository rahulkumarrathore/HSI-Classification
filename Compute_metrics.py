import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# ============================================================
# FLAGS (hardcoded -- yahin badlo)
# ============================================================

NUM_CLASSES = 9
CLASS_NAMES = [
    "Asphalt", "Meadows", "Gravel", "Trees", "Painted-metal-sheets",
    "Bare-Soil", "Bitumen", "Self-Blocking-Bricks", "Shadows",
]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """OA  = sahi pixels / total pixels
       AA  = har class ki accuracy ka average (chhoti classes ko barabar weight)
       Kappa = chance se kitna behtar (1 = perfect, 0 = random)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)      # recall per class

    oa = np.trace(cm) / cm.sum()
    present = cm.sum(axis=1) > 0                                 # jin classes ke sample hain
    aa = per_class[present].mean()
    kappa = cohen_kappa_score(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    return {
        "OA": 100 * oa,
        "AA": 100 * aa,
        "Kappa": 100 * kappa,
        "per_class": 100 * per_class,
        "cm": cm,
    }


def print_metrics(m, title=""):
    if title:
        print(f"===== {title} =====")
    print(f"OA {m['OA']:.2f}   AA {m['AA']:.2f}   Kappa {m['Kappa']:.2f}")
    for name, acc, n in zip(CLASS_NAMES, m["per_class"], m["cm"].sum(axis=1)):
        print(f"  {name:<22}{acc:7.2f}   (n={n})")


# ============================================================
# CHECKS (sirf jab ye file directly chalao)
# ============================================================

if __name__ == "__main__":
    # chhota haath se check-able example
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 0, 0]
    # class0: 3/4 = 75, class1: 2/2 = 100, class2: 2/4 = 50
    # OA = 7/10 = 70, AA = (75+100+50)/3 = 75  (baaki 6 classes ke 0 sample -> unhe AA se bahar rakhna hai)
    m = compute_metrics(y_true, y_pred)
    print_metrics(m, "toy example")
