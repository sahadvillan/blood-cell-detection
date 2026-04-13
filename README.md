# Blood Cell Detection using Deformable DETR

Object detection on the [BCCD dataset](https://github.com/Shenggan/BCCD_Dataset) using **Deformable DETR** (`SenseTime/deformable-detr`) via Hugging Face Transformers. The model detects three classes: **WBC** (white blood cells), **RBC** (red blood cells), and **Platelets**.

## Assignment Details

- **Objective**: Detect 3 classes (WBC, RBC, Platelets) using a transformer-based model.
- **Model**: [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) via Hugging Face (Transfer Learning).
- **Evaluation**: COCO mAP, Precision-Recall Curve, Confusion Matrix, 10 validation visualizations.
- **Dependency Management**: [astral-uv](https://github.com/astral-sh/uv).

## Results

| Metric | Score |
|---|---|
| mAP @ 0.50:0.95 | **51.9%** |
| mAP @ 0.50 (IoU=0.50) | **84.4%** |

Evaluation outputs (confusion matrix, PR curve, 10 validation visualizations) are saved in `eval_results/`.

## Installation

Requires Python 3.10+. Install all dependencies in one command using `uv`:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all project dependencies
uv sync
```

## Usage

### 1. Data Preparation

Download and preprocess the BCCD dataset from the official repository. This converts the VOC XML annotations to COCO JSON format and creates train/val/test splits. We additionally run a tiling script to artificially enlarge the pixel footprint of small objects (Platelets) via a 2x2 overlap grid.

```bash
uv run python preprocess.py https://github.com/Shenggan/BCCD_Dataset.git ./blood_cell_data
uv run python tile.py
```

### 2. Training

Fine-tune Deformable DETR on the BCCD dataset. Training runs for 50 epochs with Albumentations data augmentation and dataset tiling.

```bash
uv run python train.py --tiled
```

The best model checkpoint is saved to `blood_cell_model/best_model/`.

### 3. Evaluation

Run COCO mAP evaluation and generate all required deliverables (PR curve, confusion matrix, 10 validation visualizations):

```bash
uv run python eval.py --tiled --model_path blood_cell_model/best_model
```

Outputs are saved to `eval_results/`:
- `pr_curve.png` - Precision-Recall curve per class at IoU=0.50
- `confusion_matrix.png` - Confusion matrix for WBC, RBC, Platelets (+ background)
- `val_visualizations/` - 10 validation images with ground truth (green) and predictions (red)

### 4. Cloud Execution (Google Colab / Kaggle)

For GPU-accelerated training, use the provided `BCCD_DETR.ipynb` notebook. It integrates all steps above into a single interactive environment and is compatible with both Google Colab and Kaggle.

## Project Structure

```text
.
├── preprocess.py          # VOC to COCO conversion and data splitting
├── tile.py                # 2x2 overlapping grid image tiler for small objects
├── train.py               # Deformable DETR fine-tuning script
├── eval.py                # COCO mAP evaluation, PR curve, confusion matrix
├── BCCD_DETR.ipynb  # Notebook for Colab / Kaggle pipeline
├── pyproject.toml         # Project dependencies (managed by uv)
├── uv.lock                # Locked dependency versions for reproducibility
└── eval_results/          # Generated evaluation outputs
```

## Submission

- Repository is hosted on a private GitHub repository.
- Access has been granted to: `is.cemos@hs-mannheim.de`.
