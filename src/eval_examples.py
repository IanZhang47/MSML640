import os
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models.unet import UNet


class SegDataset(Dataset):
    """
    Same convention as in train.py:
    root/{images,masks}/*.png
    """
    def __init__(self, root: str):
        self.im_dir = os.path.join(root, "images")
        self.ms_dir = os.path.join(root, "masks")
        self.files = sorted([f for f in os.listdir(self.im_dir) if f.endswith(".png")])
        if not self.files:
            raise FileNotFoundError(f"No PNGs found in {self.im_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = np.array(
            Image.open(os.path.join(self.im_dir, fn)).convert("L"),
            dtype=np.float32,
        ) / 255.0
        msk = np.array(
            Image.open(os.path.join(self.ms_dir, fn)).convert("L")
        ) > 127

        x = torch.from_numpy(img).unsqueeze(0)        # 1xHxW
        y = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        return x, y, fn


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    pred, target: 1xHxW tensors (binary 0/1)
    """
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return float((2 * inter + eps) / (union + eps))


def make_overlay(img: np.ndarray,
                 pred: np.ndarray,
                 gt: np.ndarray,
                 alpha: float = 0.4) -> Image.Image:
    """
    img: HxW in [0,1]
    pred, gt: HxW binary (0/1)
    Colors:
      - prediction: red
      - ground truth: green
      - overlap: yellow
    """
    H, W = img.shape
    base = np.stack([img, img, img], axis=-1)  # HxWx3

    overlay = base.copy()

    # prediction mask (red)
    red = np.array([1.0, 0.0, 0.0])
    mask_pred = pred > 0.5
    overlay[mask_pred] = (1 - alpha) * overlay[mask_pred] + alpha * red

    # ground truth mask (green)
    green = np.array([0.0, 1.0, 0.0])
    mask_gt = gt > 0.5
    overlay[mask_gt] = (1 - alpha) * overlay[mask_gt] + alpha * green

    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    ap = argparse.ArgumentParser(
        description="Find best and worst validation slices and save overlay images."
    )
    ap.add_argument("--data-val", required=True,
                    help="Validation slice dir with images/ and masks/")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained UNet checkpoint (.pt)")
    ap.add_argument("--out", default="eval/examples",
                    help="Output dir to save PNGs")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of best slices to save")
    ap.add_argument("--bottom-k", type=int, default=5,
                    help="Number of worst slices to save")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    print("Device:", device)

    os.makedirs(args.out, exist_ok=True)
    best_dir = os.path.join(args.out, "best")
    worst_dir = os.path.join(args.out, "worst")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    # Dataset / loader
    ds = SegDataset(args.data_val)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = UNet(in_ch=1, base_ch=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    records = []  # (dice, filename, img_np, pred_np, gt_np)

    with torch.no_grad():
        for x, y, fn in tqdm(dl, desc="Val slices"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            d = dice_score(pred.cpu(), y.cpu())

            img_np = x.cpu().numpy()[0, 0]    # HxW
            pred_np = pred.cpu().numpy()[0, 0]
            gt_np = y.cpu().numpy()[0, 0]

            records.append((d, fn[0], img_np, pred_np, gt_np))

    # sort by Dice
    records.sort(key=lambda t: t[0])  # ascending
    N = len(records)
    k_top = min(args.top_k, N)
    k_bottom = min(args.bottom_k, N)

    print(f"Total val slices: {N}")
    print(f"Saving {k_top} best and {k_bottom} worst examples.")

    # worst examples
    for rank, (d, fn, img_np, pred_np, gt_np) in enumerate(records[:k_bottom], start=1):
        overlay = make_overlay(img_np, pred_np, gt_np)
        out_name = f"{rank:02d}_dice-{d:.3f}_{fn}"
        overlay.save(os.path.join(worst_dir, out_name))

    # best examples
    for rank, (d, fn, img_np, pred_np, gt_np) in enumerate(records[-k_top:], start=1):
        overlay = make_overlay(img_np, pred_np, gt_np)
        out_name = f"{rank:02d}_dice-{d:.3f}_{fn}"
        overlay.save(os.path.join(best_dir, out_name))

    print("Done. Check:")
    print("  Best examples:", best_dir)
    print("  Worst examples:", worst_dir)


if __name__ == "__main__":
    main()

