
import torch
from models.TextEmb import TextEmbedder
from typing import List, Dict
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_top1_grid(image_paths, probs_torch, cat_names, out_dir="out", cols=5, thumb_size=192):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    probs = probs_torch.detach().cpu().numpy()   # (N, K)
    pred_top1 = probs.argmax(axis=1)             # (N,)
    score_top1 = probs.max(axis=1)               # (N,)

    thumbs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB").resize((thumb_size, thumb_size))
        thumbs.append(np.asarray(im))

    N = len(thumbs)
    rows = (N + cols - 1) // cols
    plt.figure(figsize=(cols * 2.2, rows * 2.2))

    for i, (im, pred, sc) in enumerate(zip(thumbs, pred_top1, score_top1), start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"img{i}: {cat_names[int(pred)]}\nscore={float(sc):.3f}", fontsize=9)

    plt.tight_layout()
    out_path  = Path("/workspace/out/thumbs_pred_top1_prompt_ensemble.png")

    plt.savefig(out_path, dpi=160)
    plt.show()
    print(f"[saved] {out_path.resolve()}")


def plot_clip_heatmap_ic(S_ic, cat_names, imgs, out_path):
    """
    S_ic : (N_images, K_categories) numpy array or torch.Tensor (cosine-similarity or scores)
    cat_names : list[str] length K
    imgs : list of image paths (or any list; only length is used for y-ticks)
    out_path : str or Path, e.g., OUT / "clip_heatmap_ic_annot.png"
    """
    # to numpy
    if hasattr(S_ic, "detach"):
        S_ic = S_ic.detach().cpu().numpy()
    else:
        S_ic = np.asarray(S_ic)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # figure size heuristics
    W = max(6, 0.9 * len(cat_names))
    H = max(4, 0.5 * len(imgs))
    fig = plt.figure(figsize=(W, H))
    ax = plt.gca()

    im = ax.imshow(S_ic, aspect="auto")
    ax.set_title("CLIP cosine-similarity: images × categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Image")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("similarity", rotation=90)

    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, rotation=45, ha="right")
    ax.set_yticks(range(len(imgs)))
    ax.set_yticklabels([f"img{i+1}" for i in range(len(imgs))])

    # annotate values if small matrix
    if S_ic.size <= 200:
        for i in range(S_ic.shape[0]):
            for j in range(S_ic.shape[1]):
                ax.text(j, i, f"{S_ic[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.show()
    plt.close(fig)
    return out_path