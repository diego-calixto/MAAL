#!/usr/bin/env python3
"""
visualize_comparison.py
========================
For a set of sample images, loads multiple experiment checkpoints and creates
a comparison visualization showing:
    Original | baseline_pred | baseline_saliency | Attention_pred | Attention_saliency | CAM_pred | CAM_saliency | maal_expG_pred | maal_expG_saliency

Usage (from repo root):
    python scripts/visualize_comparison.py
    python scripts/visualize_comparison.py --max-images 5 --device cuda
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
    "MTLB": {
        "checkpoint": ROOT / "resultados_cluster/baseline/checkpoints/fold_0/best.pt",
        "model_type": "baseline",
    },
    "CNNAM": {
        "checkpoint": ROOT / "resultados_cluster/Attention/checkpoints/fold_0/best.pt",
        "model_type": "attention",
    },
    "CAM": {
        "checkpoint": ROOT / "resultados_cluster/CAM/checkpoints/fold_0/best.pt",
        "model_type": "cam_head",
    },
    "MAAL": {
        "checkpoint": ROOT / "resultados_cluster/maal_v3/checkpoints/fold_0/best.pt",
        "model_type": "fusion_cam",
    },
}

# Dataset directories
IMAGES_DIR = ROOT / "processed_dataset_MTL/Positive/images"
MASKS_DIR  = ROOT / "processed_dataset_MTL/Positive/masks"

# GradCAM target layer (same for all experiments)
TARGET_LAYER_NAME = "encoder.layer4[-1].conv3"

# Output directory
OUTPUT_DIR = ROOT / "outputs" / "comparison_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def unnormalize(tensor_3chw: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to an HWC uint8 numpy image."""
    img = tensor_3chw.permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


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


def apply_colormap(gray: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
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


# ── main visualization ───────────────────────────────────────────────────────

def visualize_comparison(image_paths: list[Path], device: str, threshold: float = 0.5) -> None:
    """Create a comparison visualization for all experiments."""
    
    # Load all models
    models = {}
    for exp_name, cfg in EXPERIMENTS.items():
        ckpt_path = cfg["checkpoint"]
        model_type = cfg["model_type"]
        
        print(f"\nLoading {exp_name} from {ckpt_path}")
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue
        
        model, _ = load_model_from_checkpoint(str(ckpt_path), model_type, device)
        model.eval()
        models[exp_name] = model
    
    if not models:
        print("[ERROR] No models loaded successfully.")
        return
    
    n_imgs = len(image_paths)
    exp_names = list(models.keys())
    n_exps = len(exp_names)
    
    # Layout: rows = images, cols = [Original, exp1_pred, exp1_sal, exp2_pred, exp2_sal, ...]
    n_cols = 1 + n_exps * 2
    COL_LABELS = ["Original"]
    for exp in exp_names:
        COL_LABELS.extend([f"{exp} mask", f"{exp} saliency"])
    
    fig, axes = plt.subplots(n_imgs, n_cols,
                              figsize=(n_cols * 2.5, n_imgs * 2.5),
                             squeeze=False)
    
    
    for row, img_path in enumerate(tqdm(image_paths, desc="Processing images", leave=False)):
        stem = img_path.stem
        
        # Load original image
        original_rgb = load_rgb_image(img_path)
        input_tensor = preprocess_image(original_rgb, IMAGE_SIZE)
        input_tensor = input_tensor.to(device)
        
        # Display original image
        ax = axes[row][0]
        ax.imshow(original_rgb)
        ax.axis("off")
        if row == 0:
            ax.set_title(COL_LABELS[0], fontsize=20, fontweight="bold", pad=15)
        ax.set_ylabel(stem, fontsize=8, rotation=0, labelpad=10,
                      va="center", ha="right")
        
        # Process each experiment
        for col_idx, exp_name in enumerate(exp_names):
            model = models[exp_name]
            
            # Get predicted mask and saliency map
            pred_mask_small, attn_map_small = get_predicted_mask(model, input_tensor, threshold)
            
            # Resize pred mask to originalResolution
            pred_mask = cv2.resize(
                pred_mask_small.astype(np.float32),
                (original_rgb.shape[1], original_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            
            # Get saliency map (intrinsic or GradCAM)
            if attn_map_small is not None:
                attn_map = cv2.resize(
                    attn_map_small.astype(np.float32),
                    (original_rgb.shape[1], original_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                try:
                    cam_small = compute_gradcam(model, input_tensor, device, TARGET_LAYER_NAME)
                    attn_map = resize_map(cam_small, original_rgb.shape[:2])
                except Exception as exc:
                    print(f"    [WARN] GradCAM failed for {stem} ({exp_name}): {exc}")
                    attn_map = np.zeros(original_rgb.shape[:2], dtype=np.float32)
            
            # Display predicted mask
            col_pred = 1 + col_idx * 2
            ax = axes[row][col_pred]
            ax.imshow(pred_mask, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(COL_LABELS[col_pred], fontsize=20, fontweight="bold", pad=15)
            
            # Display saliency map (translucent overlay on original)
            col_sal = 2 + col_idx * 2
            ax = axes[row][col_sal]
            cam_color = apply_colormap(attn_map, "jet")
            cam_overlay = blend_overlay(original_rgb, cam_color, alpha=0.5)
            ax.imshow(cam_overlay)
            ax.axis("off")
            if row == 0:
                ax.set_title(COL_LABELS[col_sal], fontsize=20, fontweight="bold", pad=15)
    
    plt.tight_layout(pad=0.2, h_pad=0.2)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    out_path = OUTPUT_DIR / f"experiment_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Comparison visualization for multiple experiments.")
    parser.add_argument(
        "--max-images", type=int, default=5,
        help="Maximum number of images to visualize (default: 5).",
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
    sample_images = all_images[: args.max_images]
    
    if not sample_images:
        print(f"[ERROR] No images found in {IMAGES_DIR} matching prefix '{args.image_prefix}'.")
        sys.exit(1)
    
    print(f"Using {len(sample_images)} images from {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {device}")
    print(f"Experiments: {', '.join(EXPERIMENTS.keys())}")
    
    visualize_comparison(
        image_paths=sample_images,
        device=device,
        threshold=args.threshold,
    )
    
    print(f"\nAll done. Figure saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
