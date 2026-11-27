import os

# ============================================================
# PAVIA UNIVERSITY -- DATA PATHS
# ============================================================

BASE_DIR = "/home/rahul-kumar-rathore/data/HSI/dataset"
PU_CUBE_PATH = os.path.join(BASE_DIR, "PaviaU.mat")       # key: 'paviaU'    (610, 340, 103)
PU_GT_PATH = os.path.join(BASE_DIR, "PaviaU_gt.mat")      # key: 'paviaU_gt' (610, 340)

print("pu cube", PU_CUBE_PATH, os.path.exists(PU_CUBE_PATH))
print("pu gt  ", PU_GT_PATH, os.path.exists(PU_GT_PATH))
