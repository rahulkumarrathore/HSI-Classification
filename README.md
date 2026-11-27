# HSI-Classification-SSFTT

Pixel-wise hyperspectral image classification on **Pavia University** with **SSFTT**
(Spectral–Spatial Feature Tokenization Transformer), evaluated on a **leakage-free spatial split**.

## Why a spatial split?

In HSI classification every sample is a pixel plus its P×P neighbourhood patch. With the usual
random split, a test pixel almost always sits inside the patch of some training pixel, so the
model has effectively seen the test data. On Pavia University that overlap is close to **100%**,
which is why random-split papers report ~99% OA.

This project uses a **spatial split** instead:

- For each class, pixels are sorted along the class's longer axis and cut once into
  **test | train | val** (30% / 60% / 10%).
- A buffer of `PATCH // 2` pixels around every test pixel is removed from train/val.
- Result: **0.00% test leakage**: no test pixel falls inside any training patch.

| Split | Train | Val | Test | Buffer (dropped) |
|---|---|---|---|---|
| Pixels | 23,990 | 4,153 | 12,834 | 1,799 |

## Pipeline

| File | What it does |
|---|---|
| `Dataste_directory.py` | Paths to `PaviaU.mat` / `PaviaU_gt.mat` |
| `Preprocessing.py` | Load cube, spatial split + buffer, per-band standardisation and PCA (103 → 30, fit on train pixels only), reflect padding |
| `Dataloader.py` | `PUPatchDataset`: 13×13×30 patch per pixel → `(1, 30, 13, 13)`, train/val/test loaders |
| `Models.py` | SSFTT (original `SSFTTnet.py`, unmodified) |
| `Compute_metrics.py` | OA, AA, Kappa, per-class accuracy, confusion matrix |
| `Train.py` | Training; best model selected on val OA → `runs/SSFTT_spatial_p13/best.pth` |
| `Test.py` | Test-set metrics + classification map |

All settings are hardcoded as FLAGS at the top of the file that uses them (no command-line args).

## Results (SSFTT, patch 13, PCA 30, spatial split)

| OA | AA | Kappa |
|---|---|---|
| **82.98** | **86.47** | **78.23** |

| Class | Acc. | Class | Acc. |
|---|---|---|---|
| Asphalt | 82.70 | Bare-Soil | 75.28 |
| Meadows | 77.64 | Bitumen | 100.00 |
| Gravel | 96.19 | Self-Blocking-Bricks | 98.01 |
| Trees | 96.30 | Shadows | 52.11 |
| Painted-metal-sheets | 100.00 | | |

Validation OA is ~98% while test OA is ~83%. Validation lies next to the training region, while
the test region is spatially separate. The main confusions are Meadows ↔ Bare-Soil,
Asphalt → Self-Blocking-Bricks, and Shadows → Painted-metal-sheets.

## How to run

```bash
pip install -r requirements.txt
# put PaviaU.mat and PaviaU_gt.mat in the folder set in Dataste_directory.py
# (also update the absolute paths in Dataste_directory.py, Train.py, Test.py)

python Preprocessing.py   # optional: prints split sizes, leakage %, PCA variance
python Train.py           # -> runs/SSFTT_spatial_p13/best.pth, train_log.txt
python Test.py            # -> test_result.txt, test_map.png
```

Dataset: Pavia University (ROSIS sensor, 610 × 340 px, 103 bands, 9 classes), available from the
[GIC hyperspectral datasets page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Credits

`SSFTTnet.py` is the original implementation by the SSFTT authors, used unchanged:

> L. Sun, G. Zhao, Y. Zheng, Z. Wu, "Spectral–Spatial Feature Tokenization Transformer for
> Hyperspectral Image Classification," *IEEE Transactions on Geoscience and Remote Sensing*, 2022.
> Code: https://github.com/zgr6010/HSI_SSFTT
