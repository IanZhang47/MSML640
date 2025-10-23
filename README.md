
# Patient-Friendly MRI Explainer (Pediatric Brain Tumors)

**Educational prototype** that takes an MRI volume (e.g., FLAIR), segments the *suspected* tumor region, overlays a saliency map, and generates a **non-diagnostic, plain‑English caption**. Built for clinician–caregiver communication.


## Quickstart

```bash
# 1) Create env (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Prepare slices from BraTS-like data
# Expect a folder with subject subfolders containing *_flair.nii.gz and *_seg.nii.gz.
python src/data/prepare_slices.py --root /path/to/BraTS --out data/slices --modality flair

# 3) Train a simple 2D U-Net on slices
python src/train.py --data data/slices --epochs 5 --batch-size 8 --lr 1e-3 --out checkpoints/unet

# 4) Run demo (CPU ok)
python demo/app.py
```

## Data layout (for training)
After `prepare_slices.py`, you should have:
```
data/slices/
  images/
    BraTS_0001_slice_042.png
    ...
  masks/
    BraTS_0001_slice_042.png
    ...
```

## Components
- `src/models/unet.py` — lightweight PyTorch U-Net (1-channel in → 1-channel out)
- `src/data/prepare_slices.py` — convert NIfTI volumes to PNG slices (filtering to slices with labels)
- `src/train.py` — train loop (BCE+Dice), logs to stdout
- `src/interpret/gradcam.py` — simple Grad-CAM for the last encoder block
- `src/caption/templates.py` — rule-based, non-diagnostic captions
- `demo/app.py` — Gradio UI with slice slider, overlays, captions

## Ethical note
- Strong disclaimers in UI; captions avoid diagnostic language.
- Include limitations in your report. Consider domain shift (adult → pediatric).


### BraTS-PED modality note
- The data often uses suffixes like `-t2f.nii.gz`, `-t1c.nii.gz`, `-t1n.nii.gz`, `-t2w.nii.gz`.
- `prepare_slices.py` now supports `--modality auto|t2f|flair|t2w|t1c|t1n` and matches both `*_modality.nii.gz` and `-modality.nii.gz`.
- Example for your dataset: `--modality t2f` (T2-FLAIR).


### Slice-prep tips
- Use `--limit-subjects N` to do a quick smoke test.
- Use `--dry-run` to list subjects and chosen modality without writing files.
- A per-subject summary is saved to `eval/slice_counts.csv`.
