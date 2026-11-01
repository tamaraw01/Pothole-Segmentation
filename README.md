# Pothole segmentation

A PyTorch notebook for binary road-image segmentation with UNet++ and an ImageNet-pretrained EfficientNet-B5 encoder. It covers image-mask pairing, training, validation threshold selection, and run-length encoded predictions.

The [notebook](pothole-segmentation.ipynb) contains the implementation, but no saved execution outputs or trained checkpoint. No model accuracy or Dice result is reported here.

## Pipeline

- Pair `train_<id>.jpg` images with `mask_<id>.png` masks, treating nonzero mask pixels as foreground.
- Split images by foreground-area bins, taking roughly 10% of each bin for validation with seed `42`.
- Train with a loss of `0.3 * BCE + 0.5 * Dice + 0.2 * Focal`, AdamW, cosine learning-rate decay, and CUDA automatic mixed precision when available.
- Save the checkpoint with the best validation Dice at threshold `0.5`, then search 12 thresholds from `0.25` to `0.80` on the same validation set.
- Average predictions across horizontal flips and three nominal image scales, apply morphological closing and component filtering, and export a submission CSV.

## Data and setup

Run from the repository root. The tracked `dataset.zip` uses Git LFS; a clone without downloaded LFS objects contains a pointer, not an extractable ZIP. Retrieve the object before extracting it, or supply compatible data yourself. Dataset provenance and redistribution terms are not documented in this repository; confirm them before reuse.

```text
dataset/
├── train/
│   ├── images/train_<id>.jpg
│   └── mask/mask_<id>.png
└── test/
    └── images/test_<id>.jpg
```

The notebook silently drops image-mask pairs whose masks are missing. If no pairs remain, its first-example print fails. Check pairing and readable image files before training.

Create a Python environment and install the [dependencies](requirements.txt), plus the notebook interface:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install notebook
python -m notebook pothole-segmentation.ipynb
```

Read the configuration and augmentation cells before running sequentially. The first install cell upgrades several packages independently of `requirements.txt`; skip it when managing dependencies yourself. Dependencies are not locked, and installation alone does not establish compatibility.

## Default configuration

| Setting | Value |
|---|---|
| Input canvas | 768 x 768 |
| Batch size | 8 |
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | `4e-4`, cosine decay to `4e-6` |
| Weight decay | `1e-3` |
| Encoder | `efficientnet-b5` |

A CUDA GPU is recommended for this configuration. Reduce `IMG_SIZE` and `BATCH` if memory is insufficient. CPU fallback exists, but a full training run can be slow.

## Generated files

| File | Contents |
|---|---|
| `best_model.pt` | Model state dictionary selected by validation Dice |
| `submission.csv` | `ImageId` and `rle` columns for test JPG files |

RLE uses one-based starts and column-major ordering. An empty foreground mask produces an empty string. Export cells overwrite these filenames. Validation overlays are displayed in the notebook, not saved as a report.

## Evaluation and implementation limits

- Threshold selection and checkpoint selection reuse one validation split. There is no separately labeled test evaluation or independent performance estimate in the repository.
- Validation Dice is an unweighted average of batch means, so a smaller final batch has the same weight as a full batch.
- The inference function resizes a padded square probability map directly to the original dimensions without first removing padding. Non-square inputs can therefore produce misaligned masks. Review this before using exported predictions.
- Each nominal TTA scale passes through `LongestMaxSize` to the same target size, largely canceling the intended scale change. Do not assume this implements independent multi-resolution inference.
- The `A.Affine` call uses `shift_limit`, `scale_limit`, and `rotate_limit`, which do not match the current Albumentations affine parameter names. Check warnings and effective transforms rather than assuming those settings are applied.
- No trained weights, benchmark results, deployment service, or automated test suite are included. Syntax checks are not a training or inference verification.

## Libraries

The implementation uses [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), [Albumentations](https://albumentations.ai/), and [timm](https://github.com/huggingface/pytorch-image-models).
