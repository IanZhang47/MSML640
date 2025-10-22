# src/train.py
import os, argparse
import numpy as np
from typing import Tuple
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models.unet import UNet

class SegDataset(Dataset):
    """PNG slice dataset: expects root/{images,masks}/*.png"""
    def __init__(self, root: str):
        self.im_dir = os.path.join(root, "images")
        self.ms_dir = os.path.join(root, "masks")
        self.files = sorted([f for f in os.listdir(self.im_dir) if f.endswith(".png")])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No PNGs found in {self.im_dir}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        img = np.array(Image.open(os.path.join(self.im_dir, fn)).convert("L"), dtype=np.float32) / 255.0
        msk = np.array(Image.open(os.path.join(self.ms_dir, fn)).convert("L")) > 127
        x = torch.from_numpy(img).unsqueeze(0)  # 1xHxW
        y = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        return x, y

def dice_coef(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def bce_dice_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs*target).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6
    dice = 1 - (2*inter + 1e-6)/denom
    return bce + dice.mean()

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    dices = []
    for x,y in tqdm(loader, desc="val", leave=False):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        dices.append(dice_coef(logits, y).item())
    return float(np.mean(dices)) if dices else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-train", required=True, help="Train dir with images/ and masks/")
    ap.add_argument("--data-val",   required=True, help="Val dir with images/ and masks/")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/unet_ped_t2f")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("eval", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = SegDataset(args.data_train)
    val_ds   = SegDataset(args.data_val)
    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_ch=1, base_ch=32).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    with open("eval/train_log.csv", "w") as f:
        f.write("epoch,train_loss,val_dice\n")

    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for x,y in tqdm(train_ld, desc=f"train ep{ep:02d}"):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses)) if losses else 0.0
        val_dice = evaluate(model, val_ld, device)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_dice={val_dice:.4f}")

        with open("eval/train_log.csv", "a") as f:
            f.write(f"{ep},{tr_loss:.6f},{val_dice:.6f}\n")

        if val_dice > best:
            best = val_dice
            ckpt = os.path.join(args.out, "best.pt")
            torch.save({"model": model.state_dict(),
                        "val_dice": best,
                        "epoch": ep}, ckpt)
            print(f"[saved] {ckpt} (val_dice={best:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()

