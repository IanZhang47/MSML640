import gradio as gr
import numpy as np
import torch
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
import io

from src.models.unet import UNet
from src.interpret.gradcam import SimpleSegGradCAM
from src.caption.templates import make_caption

def overlay_image(base_img, mask=None, sal=None, alpha_mask=0.4, alpha_sal=0.35):
    """Return a PIL image with overlays drawn using matplotlib."""
    fig = plt.figure(figsize=(5,5))
    plt.axis("off")
    plt.imshow(base_img, cmap="gray", interpolation="nearest")
    if mask is not None:
        plt.imshow(mask, cmap="viridis", alpha=alpha_mask, interpolation="nearest")
    if sal is not None:
        plt.imshow(sal, cmap="autumn", alpha=alpha_sal, interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def load_volume(nii_file):
    vol = nib.load(nii_file).get_fdata().astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    return vol

def predict_volume(vol, model, device):
    H, W = 256, 256
    slices, masks, salmaps = [], [], []
    model.eval()
    cam = SimpleSegGradCAM(model, model.enc4[3])
    for k in range(vol.shape[2]):
        img = vol[:, :, k]
        img_r = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((H, W), Image.BILINEAR)) / 255.0
        x = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.enable_grad():
            x.requires_grad_(True)
            cam_map, probs = cam(x)

        pred = probs.detach().cpu().squeeze().numpy()
        sal  = cam_map.detach().cpu().squeeze().numpy()
        slices.append(img_r); masks.append(pred); salmaps.append(sal)
    cam.close()
    return np.stack(slices), np.stack(masks), np.stack(salmaps)


def start_session(nii_vol, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, base_ch=32).to(device)
    if checkpoint is not None:
        sd = torch.load(checkpoint.name, map_location=device)
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)
    vol = load_volume(nii_vol.name)
    slices, masks, sal = predict_volume(vol, model, device)
    state = {
        "slices": slices,
        "masks": masks,
        "sal": sal,
        "idx": 0
    }
    img = overlay_image(slices[0], masks[0], sal[0])
    caption = make_caption(slices[0], masks[0], sal[0])
    return state, img, caption

def update_idx(state, idx):
    idx = int(idx)
    state["idx"] = idx
    img = overlay_image(state["slices"][idx], state["masks"][idx], state["sal"][idx])
    caption = make_caption(state["slices"][idx], state["masks"][idx], state["sal"][idx])
    return state, img, caption

with gr.Blocks() as demo:
    gr.Markdown("## Patient-Friendly MRI Explainer")
    gr.Markdown("Upload a single-modality MRI volume (e.g., FLAIR `.nii`/`.nii.gz`) and an optional trained checkpoint. This prototype overlays a simple segmentation and saliency map, then generates a neutral caption.")
    with gr.Row():
        vol = gr.File(label="MRI volume (.nii or .nii.gz)")
        ckpt = gr.File(label="Checkpoint (optional, .pt)")
    btn = gr.Button("Run")
    state = gr.State()
    slider = gr.Slider(0, 0, value=0, step=1, label="Slice index (auto-ranged after load)")
    out_img = gr.Image(label="Slice + overlays", type="pil")
    out_cap = gr.Textbox(label="Caption", lines=3)

    def _run(vol_file, ckpt_file):
        st, img, cap = start_session(vol_file, ckpt_file)
        slider.maximum = int(st["slices"].shape[0]-1)
        slider.value = 0
        return st, gr.update(maximum=slider.maximum, value=0), img, cap

    btn.click(_run, inputs=[vol, ckpt], outputs=[state, slider, out_img, out_cap])
    slider.release(update_idx, inputs=[state, slider], outputs=[state, out_img, out_cap])

if __name__ == "__main__":
    demo.launch()
