import numpy as np

def centroid(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)

def side_from_centroid(cx, width):
    if cx is None: return "center"
    return "left" if cx < width/2 else "right"

def make_caption(img, mask, saliency=None):
    """Generate a neutral, non-diagnostic caption from a slice.
    Args:
        img: HxW float array 0..1
        mask: HxW float array 0..1 (predicted tumor probability)
        saliency: optional HxW float array 0..1
    """
    H, W = mask.shape
    area = float((mask > 0.5).sum()) / (H*W + 1e-6)
    cx_cy = centroid(mask)
    side = side_from_centroid(cx_cy[0], W) if cx_cy else "center"

    size_phrase = (
        "small highlighted area"
        if area < 0.02 else
        "moderate highlighted area"
        if area < 0.08 else
        "larger highlighted area"
    )
    sal_phrase = ""
    if saliency is not None:
        top_q = (saliency > 0.7).sum() / (H*W + 1e-6)
        if top_q > 0.02:
            sal_phrase = " The heatmap shows regions the model found most influential."
    cap = (
        f"This slice shows a {size_phrase} on the {side} side. "
        f"The overlay marks tissue the model considers unusual.{sal_phrase} "
    )
    return cap
