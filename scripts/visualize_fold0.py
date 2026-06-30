#!/usr/bin/env python3
"""
visualize_fold0.py
==================
For each experiment in resultados_cluster, loads the fold-0 best checkpoint,
runs inference on a set of sample images, computes GradCAM, and saves a
multi-panel figure per experiment showing:
    Original Image | Ground-Truth Mask | Predicted Mask | GradCAM

Usage (from repo root):
    python scripts/visualize_fold0.py
    python scripts/visualize_fold0.py --max-images 5 --device cuda
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.saliency.utils import (
    load_model_from_checkpoint,
    load_rgb_image,
    preprocess_image,
    resize_map,
    normalize_map,
    resolve_target_layer,
)
from src.saliency.methods import GradCAM
from src.utils.common import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# ── experiment registry ──────────────────────────────────────────────────────
EXPERIMENTS = {
    "baseline": {
        "checkpoint": ROOT / "resultados_cluster/baseline/checkpoints/fold_0/best.pt",
        "model_type": "baseline",
    },
    "Attention": {
        "checkpoint": ROOT / "resultados_cluster/Attention/checkpoints/fold_0/best.pt",
        "model_type": "attention",
    },
    "CAM": {
        "checkpoint": ROOT / "resultados_cluster/CAM/checkpoints/fold_0/best.pt",
        "model_type": "cam_head",
    },
    "Fusion_CAM": {
        "checkpoint": ROOT / "resultados_cluster/Fusion_CAM/checkpoints/fold_0/best.pt",
        "model_type": "fusion_cam",
    },
    "MAAL": {
        "checkpoint": ROOT / "resultados_cluster/MAAL/checkpoints/fold_0/best.pt",
        "model_type": "maal",
    },
    "MAAL_V2": {
        "checkpoint": ROOT / "resultados_cluster/MAAL_V2/resultados_cluster/checkpoints/fold_0/best.pt",
        "model_type": "maal",
    },
    "MAAL_G": {
        "checkpoint": ROOT / "resultados_cluster/maal_v3/checkpoints/fold_0/best.pt",
        "model_type": "maal_g",
    },
    "MAAL_H":
    {
        "checkpoint": ROOT / "resultados_cluster/maal_v4/checkpoints/fold_0/best.pt",
        "model_type": "maal_h",
    },
}

# Dataset directories
IMAGES_DIR = ROOT / "processed_dataset_MTL/Positive/images"
MASKS_DIR  = ROOT / "processed_dataset_MTL/Positive/masks"

# GradCAM target layer (same for all experiments)
TARGET_LAYER_NAME = "encoder.layer4[-1].conv3"

# Output directory
OUTPUT_DIR = ROOT / "outputs" / "fold0_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def unnormalize(tensor_3chw: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to an HWC uint8 numpy image."""
    img = tensor_3chw.permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def load_gt_mask(stem: str) -> np.ndarray | None:
    """Return the ground-truth mask as a float32 [0,1] numpy array or None."""
    for ext in (".png", ".jpg"):
        p = MASKS_DIR / (stem + ext)
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m.astype(np.float32) / 255.0
    return None


def get_predicted_mask(model, input_tensor: torch.Tensor, threshold: float = 0.5) -> tuple:
    """Run inference and return binary predicted mask and attention map as float32 [0,1]."""
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Extract segmentation output
    y_seg = outputs[1] if len(outputs) > 1 else outputs[0]
    prob  = torch.sigmoid(y_seg)                        # [1,1,H,W]
    mask  = (prob > threshold).float()
    
    # Extract intrinsic interpretation map based on model type
    attn_map = None
    
    if len(outputs) == 4:
        # attention.py: returns (y_cls, y_seg, attention_logits, attention_map)
        # attention_map is already sigmoid-ed
        if isinstance(outputs[3], torch.Tensor):
            attn_map = outputs[3]  # attention_map (already sigmoid-ed)
        # maal_expG.py: returns (y_cls, y_seg, fused, saliency_maps)
        # saliency_maps is a list, fused (outputs[2]) needs sigmoid
        elif isinstance(outputs[3], list):
            attn_map = torch.sigmoid(outputs[2])  # fused map
    elif len(outputs) == 3:
        # cam_head.py: returns (y_cls, y_seg, cam_logits)
        # cam_logits needs sigmoid
        if isinstance(outputs[2], torch.Tensor):
            attn_map = torch.sigmoid(outputs[2])  # cam_logits
    
    if attn_map is not None:
        attn_map = attn_map.squeeze().cpu().numpy()  # [H,W]
    
    return mask.squeeze().cpu().numpy(), attn_map       # [H,W], [H,W] or None


def compute_gradcam(model, input_tensor: torch.Tensor, device: str,
                    target_layer_name: str) -> np.ndarray:
    """Compute Grad-CAM and return a [0,1] float32 heatmap at input resolution."""
    target_layer = resolve_target_layer(model, target_layer_name)
    explainer = GradCAM(
        model=model,
        target_layer=target_layer,
        device=device,
        target_type="soft",
        threshold=0.5,
    )
    cam = explainer.generate(input_tensor)           # normalised [0,1] numpy
    return cam.astype(np.float32)


def apply_colormap(gray: np.ndarray, cmap_name: str = "rainbow_r") -> np.ndarray:
    """Convert a [0,1] float map to a colour image (H,W,3) uint8."""
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(gray)                                # (H,W,4) float
    return (rgba[..., :3] * 255).astype(np.uint8)


def blend_overlay(image_rgb: np.ndarray, heatmap_rgb: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    hm  = heatmap_rgb.astype(np.float32)
    blended = np.clip(img * (1 - alpha) + hm * alpha, 0, 255).astype(np.uint8)
    return blended


# ── per-experiment visualisation ─────────────────────────────────────────────

def visualize_experiment(exp_name: str, cfg: dict, image_paths: list[Path],
                         device: str, threshold: float = 0.5, save_heatmap: bool = False) -> None:
    ckpt_path = cfg["checkpoint"]
    model_type = cfg["model_type"]

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Model type : {model_type}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return

    # Load model
    model, _ = load_model_from_checkpoint(str(ckpt_path), model_type, device)
    model.eval()
    # Prepare heatmap output directory if needed
    if save_heatmap:
        heatmap_dir = OUTPUT_DIR / f"{exp_name}_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    n_imgs = len(image_paths)
    # Layout: rows = images, cols = [Original, GT Mask, Pred Mask, Attention/GradCAM Overlay]
    COL_LABELS = ["Original", "Ground Truth", "Predicted Mask", "Attention/GradCAM"]
    n_cols = len(COL_LABELS)

    fig, axes = plt.subplots(n_imgs, n_cols,
                             figsize=(n_cols * 3.5, n_imgs * 3.5),
                             squeeze=False)

    fig.suptitle(
        f"Experiment: {exp_name}  –  Fold 0  (threshold={threshold})",
        fontsize=16, fontweight="bold", y=1.01,
    )

    for row, img_path in enumerate(tqdm(image_paths, desc=f"  {exp_name}", leave=False)):
        stem = img_path.stem           # e.g. "001_0_1152"

        # ── Load original image ──────────────────────────────────────────
        original_rgb = load_rgb_image(img_path)               # HxWx3 uint8
        input_tensor = preprocess_image(original_rgb, IMAGE_SIZE)
        input_tensor = input_tensor.to(device)

        # ── Ground-truth mask ────────────────────────────────────────────
        gt_mask = load_gt_mask(stem)
        if gt_mask is None:
            gt_mask = np.zeros(original_rgb.shape[:2], dtype=np.float32)

        # ── Predicted mask and intrinsic attention ───────────────────────────────
        pred_mask_small, attn_map_small = get_predicted_mask(model, input_tensor, threshold)
        # Resize pred from model resolution → original resolution
        pred_mask = cv2.resize(
            pred_mask_small.astype(np.float32),
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # ── Intrinsic attention map or GradCAM ────────────────────────────────────
        if attn_map_small is not None:
            # Use intrinsic attention map from model
            attn_map = cv2.resize(
            attn_map_small.astype(np.float32),
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        cam_label = "Intrinsic Attention"
        if save_heatmap:
            # Save raw heatmap (grayscale)
            raw_path = heatmap_dir / f"{stem}_heatmap_raw.png"
            cv2.imwrite(str(raw_path), (attn_map * 255).astype(np.uint8))
            # Save colored heatmap
            colored = apply_colormap(attn_map, "rainbow_r")
            color_path = heatmap_dir / f"{stem}_heatmap_colored.png"
            cv2.imwrite(str(color_path), colored)
        else:
            # Fall back to GradCAM for models without intrinsic attention
            try:
                cam_small = compute_gradcam(model, input_tensor, device, TARGET_LAYER_NAME)
                attn_map = resize_map(cam_small, original_rgb.shape[:2])  # → original resolution
                cam_label = "GradCAM"
                if save_heatmap:
                    raw_path = heatmap_dir / f"{stem}_gradcam_raw.png"
                    cv2.imwrite(str(raw_path), (attn_map * 255).astype(np.uint8))
                    colored = apply_colormap(attn_map, "rainbow_r")
                    color_path = heatmap_dir / f"{stem}_gradcam_colored.png"
                    cv2.imwrite(str(color_path), colored)
            except Exception as exc:
                print(f"    [WARN] GradCAM failed for {stem}: {exc}")
                attn_map = np.zeros(original_rgb.shape[:2], dtype=np.float32)
                cam_label = "GradCAM (failed)"

        cam_color   = apply_colormap(attn_map, "rainbow_r")
        cam_overlay = blend_overlay(original_rgb, cam_color, alpha=0.45)
        if save_heatmap:
            overlay_path = heatmap_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(overlay_path), cam_overlay)

        # ── Resize GT mask to original image size ─────────────────────────
        if gt_mask.shape != original_rgb.shape[:2]:
            gt_mask = cv2.resize(
                gt_mask.astype(np.float32),
                (original_rgb.shape[1], original_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # ── Draw cells ───────────────────────────────────────────────────
        cells = [original_rgb, gt_mask, pred_mask, cam_overlay]
        cmaps = [None, "gray", "gray", None]

        for col, (cell, cmap) in enumerate(zip(cells, cmaps)):
            ax = axes[row][col]
            ax.imshow(cell, cmap=cmap)
            ax.axis("off")
            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=12, fontweight="bold", pad=6)
            if col == 0:
                # Show image stem as row label
                ax.set_ylabel(stem, fontsize=8, rotation=0, labelpad=80,
                              va="center", ha="right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{exp_name}_fold0_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fold-0 visualization for all experiments.")
    parser.add_argument(
        "--max-images", type=int, default=10,
        help="Maximum number of images to visualize per experiment (default: 10).",
    )
    parser.add_argument(
        "--image-prefix", type=str, default="001",
        help='Only use images whose filename starts with this prefix (default: "001").',
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Torch device, e.g. "cpu" or "cuda" (default: cpu).',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid threshold for binarising the segmentation output (default: 0.5).",
    )
    parser.add_argument(
        "--save-heatmap", action="store_true",
        help="Save the saliency/attention heatmap as a separate image per input.",
    )
    parser.add_argument(
        "--image-path", type=str, default=None,
        help="Path to a specific image file to process (overrides max-images and prefix).",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Subset of experiments to run (default: all). E.g. --experiments baseline Attention",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Choose device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available – falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # Collect sample images
    all_images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if args.image_prefix:
        all_images = [p for p in all_images if p.stem.startswith(args.image_prefix)]
    # If a specific image path is provided, use only that image
    if args.image_path:
        specific_path = Path(args.image_path)
        if not specific_path.exists():
            print(f"[ERROR] Specified image not found: {specific_path}")
            sys.exit(1)
        sample_images = [specific_path]
    else:
        sample_images = all_images[: args.max_images]

    if not sample_images:
        print(f"[ERROR] No images found in {IMAGES_DIR} matching prefix '{args.image_prefix}'.")
        sys.exit(1)

    print(f"Using {len(sample_images)} images from {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {device}")

    experiments = args.experiments or list(EXPERIMENTS.keys())

    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"[WARN] Unknown experiment '{exp_name}' – skipping.")
            continue
        visualize_experiment(
            exp_name=exp_name,
            cfg=EXPERIMENTS[exp_name],
            image_paths=sample_images,
            device=device,
            threshold=args.threshold,
            save_heatmap=args.save_heatmap,
        )

    print(f"\nAll done. Figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
