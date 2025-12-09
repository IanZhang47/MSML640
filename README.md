# MRI Explainer for Pediatric Brain Tumors

This project takes a pediatric brain MRI volume and produces:

- A 2D tumor segmentation (slice-wise U-Net)
- A saliency heatmap (Grad-CAM)
- A simple, plain-English caption for each slice

It is implemented in PyTorch and provides a Gradio demo UI for interactive exploration.

---

## 1. Project Overview

### Goal

Build a small but complete pipeline for:

- Converting 3D BraTS-PED NIfTI volumes into 2D training slices
- Training a 2D U-Net for whole-tumor segmentation
- Visualizing model focus with Grad-CAM
- Generating short, human-readable captions tied to the overlays

The primary audience is pediatric brain tumor imaging, but the pipeline is general enough to adapt to other volumetric segmentation tasks.

---

## 2. Repository Structure

```text
mri-explainer/
  ├─ src/
  │   ├─ models/
  │   │   └─ unet.py            # 2D U-Net in PyTorch
  │   ├─ data/
  │   │   └─ prepare_slices.py  # NIfTI → 2D PNG slices
  │   ├─ interpret/
  │   │   └─ gradcam.py         # Simple Grad-CAM for segmentation
  │   ├─ caption/
  │   │   └─ templates.py       # Simple rule-based caption generator
  │   └─ train.py               # Training script (train/val)
  ├─ demo/
  │   └─ app.py                 # Gradio UI
  ├─ data/                      # Local data (not in git)
  ├─ checkpoints/               # Saved model checkpoints
  ├─ eval/                      # Logs, CSV summaries, plots
  ├─ requirements.txt
  ├─ LICENSE
  └─ README.md
````

---

## 3. Environment Setup

Tested with:

* Python 3.10+
* PyTorch 2.x (CPU mode)
* Ubuntu/Linux

```bash
cd mri-explainer

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Data (BraTS-PED)

Assumes the ASNR-MICCAI BraTS 2023 Pediatric dataset, with structure like:

```text
ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/
  BraTS-PED-00002-000/
    BraTS-PED-00002-000-t2f.nii.gz
    BraTS-PED-00002-000-t2w.nii.gz
    BraTS-PED-00002-000-t1c.nii.gz
    BraTS-PED-00002-000-t1n.nii.gz
    BraTS-PED-00002-000-seg.nii.gz
  ...
```

* Training split has `*seg.nii.gz` labels.
* Validation split (official “ValidationData”) is typically unlabeled and can be used for qualitative testing.

By default, the pipeline uses **T2-F** / **T2-FLAIR** (`*-t2f.nii.gz`) as the main input modality.

---

## 5. Slice Preparation

Use `prepare_slices.py` to convert volumes to 2D PNG slices for supervised training.

Basic command (full training data):

```bash
python src/data/prepare_slices.py \
  --root /path/to/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData \
  --out  data/slices_ped_t2f_full \
  --modality t2f \
  --axis 2 \
  --min-area 1
```

This produces:

```text
data/slices_ped_t2f_full/
  images/
    BraTS-PED-00002-000_slice_042.png
    ...
  masks/
    BraTS-PED-00002-000_slice_042.png
    ...
```

Key flags:

* `--modality`: `t2f | flair | t2w | t1c | t1n | auto`
* `--axis`: slicing axis (0, 1, or 2)
* `--min-area`: minimum tumor pixels per slice
* `--size`: output image size (default 256×256)
* `--limit-subjects`: restrict to first N subjects (smoke tests)
* `--dry-run`: do not write PNGs; just report selected modalities
* `--summary-csv`: per-subject statistics (default `eval/slice_counts.csv`)

Example dry-run:

```bash
python src/data/prepare_slices.py \
  --root /path/to/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData \
  --out  data/slices_ped_t2f_full \
  --modality t2f \
  --limit-subjects 3 \
  --dry-run
```

---

## 6. Train / Validation Split

For quantitative evaluation, a **subject-level split** from the training set is used. Typical approach:

1. Randomly sample a subset of subjects for validation (e.g., 20–25).
2. Create two subject-root folders:

   * `data/roots/train_subj/`
   * `data/roots/val_subj/`
3. Symlink or copy subject folders into each root.
4. Run `prepare_slices.py` separately on each root:

```bash
# Train slices
python src/data/prepare_slices.py \
  --root data/roots/train_subj \
  --out  data/slices_ped_t2f_train \
  --modality t2f --axis 2 --min-area 1

# Val slices
python src/data/prepare_slices.py \
  --root data/roots/val_subj \
  --out  data/slices_ped_t2f_val \
  --modality t2f --axis 2 --min-area 1
```

---

## 7. Training

Use `src/train.py` to train a 2D U-Net on the slice dataset.

```bash
python -m src.train \
  --data-train data/slices_ped_t2f_train \
  --data-val   data/slices_ped_t2f_val \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-3 \
  --out checkpoints/unet_ped_t2f
```

Outputs:

* `checkpoints/unet_ped_t2f/best.pt` — model weights and metadata
* `eval/train_log.csv` — per-epoch train loss and validation Dice

You can also create a separate config to train on all slices (`data/slices_ped_t2f_full`) once you are satisfied with the validation behaviour.

---

## 8. Gradio Demo

The demo provides an interactive interface for:

* Uploading a NIfTI volume
* Running the trained model slice-wise
* Viewing segmentation and Grad-CAM
* Reading the generated caption

Run from the repository root:

```bash
python -m demo.app
```

Then open the printed local URL

In the UI:

* Upload one `*.nii` / `*.nii.gz` volume (e.g. a `*-t2f.nii.gz` file).
* Optionally upload a checkpoint (`.pt`) such as `checkpoints/unet_ped_t2f/best.pt`.
* Use the slice slider to navigate the volume.

---

## 9. Evaluation & Analysis

Typical checks:

* **Quantitative (on labeled val split):**

  * Mean/median Dice score for whole-tumor segmentation
  * Distribution of Dice vs. slice index or subject

* **Qualitative:**

  * Visual inspection of segmentation + heatmap overlays
  * Captions that roughly correspond to the visible highlighted region
  * Side (left/right) and size descriptors that make sense across slices



