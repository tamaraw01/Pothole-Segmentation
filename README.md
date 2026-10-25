# 🕳️ Pothole Segmentation

**Semantic segmentation** of road potholes using **UNet++ with EfficientNet-B5** encoder, trained with mixed-precision and strong augmentations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

---

## 📋 Overview

This project trains a semantic segmentation model to detect and delineate potholes in road images. The pipeline is implemented in a single Jupyter notebook and includes:

- **Model:** UNet++ decoder with EfficientNet-B5 backbone (ImageNet pre-trained)
- **Loss:** Weighted combination of BCE (0.3) + Dice (0.5) + Focal (0.2)
- **Training:** Mixed-precision (AMP), cosine annealing LR, stratified train/val split
- **Inference:** Multi-scale test-time augmentation (TTA) + morphological post-processing
- **Output:** RLE-encoded submission CSV

## 📁 Project Structure

```
Pothole Segmentation/
├── pothole-segmentation.ipynb   # Main training & inference notebook
├── dataset/
│   ├── train/
│   │   ├── images/              # 498 training images (train_XXX.jpg)
│   │   └── mask/                # 498 binary masks   (mask_XXX.png)
│   └── test/
│       └── images/              # 295 test images    (test_XXX.jpg)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tamaraw01/Pothole-Segmentation.git
cd Pothole-Segmentation
```

### 2. Set up the dataset

Place the extracted dataset under `dataset/` following the structure above. The [notebook](pothole-segmentation.ipynb) uses paths relative to the project root, so launch Jupyter from `Pothole-Segmentation/`.

Training image names must end with an underscore and an identifier, such as `train_326.jpg`. The notebook pairs that image with `dataset/train/mask/mask_326.png` and drops pairs whose mask file is missing. If no pairs remain, the data preparation cell fails when it prints the first pair. Check the directory names and image/mask identifiers before running it.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook pothole-segmentation.ipynb
```

Execute cells sequentially. Training runs for **25 epochs** with mixed-precision on GPU.

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Input size | 768 × 768 |
| Batch size | 8 |
| Epochs | 25 |
| Learning rate | 4e-4 → 4e-6 (cosine) |
| Weight decay | 1e-3 |
| Optimizer | AdamW |
| Encoder | EfficientNet-B5 |

> 💡 **OOM?** Reduce `IMG_SIZE` to 512 and `BATCH` to 4 in the config cell.

## 📊 Pipeline Highlights

### Augmentations (Albumentations)
- Affine (shift/scale/rotate)
- Brightness/contrast & HSV jitter
- Gaussian blur, motion blur, Gaussian noise
- Coarse dropout (cutout-style)

### Test-Time Augmentation
- 3 scales: 0.85×, 1.0×, 1.15×
- Horizontal flip per scale
- Averaged probability maps

### Post-Processing
- Morphological closing (5×5 kernel)
- Connected component filtering (min area = 200 px)

## 🙏 Acknowledgments

- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [timm](https://github.com/huggingface/pytorch-image-models)
