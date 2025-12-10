import os
import sys

# Ensure project root is on sys.path when running as "python demo/app.py"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
import numpy as np
import torch
from PIL import Image
import nibabel as nib

from src.models.unet import UNet
from src.interpret.gradcam import SimpleSegGradCAM
from src.caption.templates import make_caption


# ------------------------- Overlay utilities ------------------------- #

def overlay_image(base_img, pred_mask=None, gt_mask=None,
                  alpha_pred=0.4, alpha_gt=0.4):
    """
    Create a gray + red/green/yellow overlay similar to eval_examples.py.

    - base_img: 2D float array in [0, 1] (MRI slice)
    - pred_mask: 2D float/binary array (model prediction)
    - gt_mask: 2D float/binary array (ground-truth; optional)
    Colors:
      - prediction: red
      - ground truth: green
      - overlap: yellow (red + green)
    """
    base_img = np.asarray(base_img, dtype=np.float32)
    if base_img.max() > 1.5:
        base_img = base_img / 255.0

    H, W = base_img.shape
    base = np.stack([base_img, base_img, base_img], axis=-1)  # HxWx3
    overlay = base.copy()

    # Ground truth in green
    if gt_mask is not None:
        g = np.asarray(gt_mask, dtype=np.float32)
        if g.shape != (H, W):
            g = np.array(
                Image.fromarray(g).resize((W, H), Image.NEAREST)
            )
        mask_gt = g > 0.5
        green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        overlay[mask_gt] = (1 - alpha_gt) * overlay[mask_gt] + alpha_gt * green

    # Prediction in red
    if pred_mask is not None:
        p = np.asarray(pred_mask, dtype=np.float32)
        if p.shape != (H, W):
            p = np.array(
                Image.fromarray(p).resize((W, H), Image.NEAREST)
            )
        mask_pred = p > 0.5
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay[mask_pred] = (1 - alpha_pred) * overlay[mask_pred] + alpha_pred * red

    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ------------------------- Volume loading ------------------------- #

def load_volume(nii_path: str):
    """Load and normalize a NIfTI volume to [0,1]."""
    vol = nib.load(nii_path).get_fdata().astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    return vol


def load_seg_for_volume(nii_path: str):
    """
    Try to find a BraTS-style segmentation file next to the given MRI volume.
    E.g., for BraTS-PED-00016-000-t2f.nii.gz, look for BraTS-PED-00016-000-seg.nii.gz.
    Returns a float32 volume with binary mask (0/1), or None if not found.
    """
    dir_ = os.path.dirname(nii_path)
    base = os.path.basename(nii_path)  # e.g., BraTS-PED-00016-000-t2f.nii.gz
    name = base.split(".nii")[0]       # strip .nii or .nii.gz

    # Strip common modality suffixes
    subj_prefix = name
    for suf in ("-t2f", "-t2w", "-t1c", "-t1n", "-flair"):
        if name.endswith(suf):
            subj_prefix = name[:-len(suf)]
            break

    candidates = [
        os.path.join(dir_, subj_prefix + "-seg.nii.gz"),
        os.path.join(dir_, subj_prefix + "-seg.nii"),
        os.path.join(dir_, subj_prefix + "_seg.nii.gz"),
        os.path.join(dir_, subj_prefix + "_seg.nii"),
    ]
    for c in candidates:
        if os.path.exists(c):
            seg = nib.load(c).get_fdata().astype(np.float32)
            # BraTS labels: 0,1,2,4; convert to binary (non-zero = tumor)
            seg = (seg > 0).astype(np.float32)
            return seg

    return None


# ------------------------- Prediction over volume ------------------------- #

def predict_volume(vol, seg_vol, model, device, axis: int = 2):
    """
    Run model + Grad-CAM over all slices along the given axis.
    Returns:
      slices: [N, H, W] float in [0,1]
      masks:  [N, H, W] float probs in [0,1]
      sal:    [N, H, W] float saliency in [0,1]
      gts:    [N, H, W] float binary GT masks (0/1); zeros if seg_vol is None
    """
    H, W = 256, 256
    slices, masks, salmaps, gts = [], [], [], []

    model.eval()
    cam = SimpleSegGradCAM(model, model.enc4[3])

    # Ensure we slice along axis 2 (axial); adapt if you want axis switching
    assert axis == 2, "predict_volume currently assumes axis=2 (axial)."

    for k in range(vol.shape[axis]):
        img = vol[:, :, k]

        # Prepare image
        img_r = np.array(
            Image.fromarray((img * 255).astype(np.uint8)).resize((H, W), Image.BILINEAR)
        ) / 255.0
        x = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).float().to(device)

        # GT slice if available
        if seg_vol is not None and seg_vol.shape == vol.shape:
            gt_raw = seg_vol[:, :, k]
            gt_r = np.array(
                Image.fromarray(gt_raw.astype(np.float32)).resize((H, W), Image.NEAREST)
            )
            gt_mask = (gt_r > 0).astype(np.float32)
        else:
            gt_mask = np.zeros((H, W), dtype=np.float32)

        # Grad-CAM + prediction
        with torch.enable_grad():
            x.requires_grad_(True)
            cam_map, probs = cam(x)

        pred = probs.detach().cpu().squeeze().numpy()
        sal = cam_map.detach().cpu().squeeze().numpy()

        slices.append(img_r)
        masks.append(pred)
        salmaps.append(sal)
        gts.append(gt_mask)

    cam.close()
    return (
        np.stack(slices),
        np.stack(masks),
        np.stack(salmaps),
        np.stack(gts),
    )


# ------------------------- Session helpers ------------------------- #

def start_session(nii_vol, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, base_ch=32).to(device)

    if checkpoint is not None:
        sd = torch.load(checkpoint.name, map_location=device)
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)

    vol_path = nii_vol.name
    vol = load_volume(vol_path)
    seg_vol = load_seg_for_volume(vol_path)  # may be None

    slices, masks, sal, gts = predict_volume(vol, seg_vol, model, device)

    state = {
        "slices": slices,
        "masks": masks,
        "sal": sal,
        "gts": gts,
        "idx": 0,
    }

    idx = 0
    img = overlay_image(slices[idx], masks[idx], gts[idx])
    caption = make_caption(slices[idx], masks[idx], sal[idx])

    return state, img, caption


def update_idx(state, idx):
    idx = int(idx)
    idx = max(0, min(idx, state["slices"].shape[0] - 1))
    state["idx"] = idx

    img = overlay_image(
        state["slices"][idx],
        state["masks"][idx],
        state["gts"][idx],
    )
    caption = make_caption(
        state["slices"][idx],
        state["masks"][idx],
        state["sal"][idx],
    )
    return state, img, caption


# ------------------------- Gradio UI ------------------------- #

with gr.Blocks() as demo:
    gr.Markdown("## MRI Explainer")
    gr.Markdown(
        "Upload a single-modality MRI volume (e.g., T2-FLAIR `.nii`/`.nii.gz`) and an optional trained checkpoint. "
        "If a BraTS-style segmentation file (e.g., `*-seg.nii.gz`) is present in the same folder, "
        "the overlay will show prediction (red) vs. ground truth (green, overlap in yellow)."
    )

    with gr.Row():
        vol = gr.File(label="MRI volume (.nii or .nii.gz)")
        ckpt = gr.File(label="Checkpoint (optional, .pt)")

    btn = gr.Button("Run")
    state = gr.State()

    slider = gr.Slider(
        minimum=0,
        maximum=0,
        value=0,
        step=1,
        label="Slice index (auto-ranged after load)",
    )

    out_img = gr.Image(label="Slice + overlay", type="pil")
    out_cap = gr.Textbox(label="Caption", lines=3)

    def _run(vol_file, ckpt_file):
        st, img, cap = start_session(vol_file, ckpt_file)
        max_idx = int(st["slices"].shape[0] - 1)
        return (
            st,
            gr.update(maximum=max_idx, value=0),
            img,
            cap,
        )

    btn.click(
        _run,
        inputs=[vol, ckpt],
        outputs=[state, slider, out_img, out_cap],
    )

    slider.release(
        update_idx,
        inputs=[state, slider],
        outputs=[state, out_img, out_cap],
    )


if __name__ == "__main__":
    demo.launch()

