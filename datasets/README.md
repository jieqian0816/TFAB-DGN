# Datasets

This directory is intentionally kept empty in the repository. The raw
hyperspectral datasets (`.mat` / `.npy`) are **not** versioned because they are
large binary files.

## How to obtain the datasets

Download the four public hyperspectral datasets and place them under the
corresponding sub-folders:

| Dataset | Folder | Required files |
|---|---|---|
| Indian Pines (IP) | `datasets/Indian_pines/` | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |
| Pavia University (PU) | `datasets/PaviaU/` | `PaviaU.mat`, `PaviaU_gt.mat` |
| Salinas (SA) | `datasets/Salinas/` | `Salinas_corrected.mat`, `Salinas_gt.mat` |
| Kennedy Space Center (LK) | `datasets/KSC/` | `KSC.mat`, `KSC_gt.mat` |

The data can be downloaded from the official sources:

- Indian Pines & Salinas: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- Pavia University: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- Kennedy Space Center: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

After downloading, the directory layout should look like:

```
datasets/
├── Indian_pines/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── Salinas/
│   ├── Salinas_corrected.mat
│   └── Salinas_gt.mat
└── KSC/
    ├── KSC.mat
    └── KSC_gt.mat
```
