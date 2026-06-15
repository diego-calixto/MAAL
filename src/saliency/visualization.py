from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.saliency.config import HEATMAP_DIR, OVERLAY_DIR, RAW_DIR


def save_gray_image(output_path: Path, image: np.ndarray) -> None:
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(output_path), image_uint8)


def create_heatmap(saliency_map: np.ndarray, cmap: str = 'jet') -> np.ndarray:
    # Accept either 2D maps or 3D maps (H,W,3) / (H,W,4). Collapse to single channel if needed.
    if saliency_map.ndim == 3:
        # If already RGB/RGBA, convert to grayscale luminance
        if saliency_map.shape[2] in (3, 4):
            saliency_map = saliency_map[..., :3]
            saliency_map = np.dot(saliency_map, [0.2989, 0.5870, 0.1140])
        else:
            saliency_map = np.squeeze(saliency_map)

    cmap_out = plt.get_cmap(cmap)(saliency_map)
    heatmap = cmap_out[:, :, :3]
    return (heatmap * 255.0).astype(np.uint8)


def create_overlay(original_image: np.ndarray, saliency_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # Ensure heatmap creation handles channel collapsing; keep original in RGB
    original = original_image.astype(np.float32) / 255.0
    heatmap = create_heatmap(saliency_map).astype(np.float32) / 255.0
    if heatmap.shape[:2] != original.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = (original * (1.0 - alpha)) + (heatmap * alpha)
    return (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_saliency_outputs(
    image_path: Path,
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> None:
    output_name = image_path.stem
    raw_path = output_dir / 'raw' / f'{output_name}_{prefix}_raw.png'
    heatmap_path = output_dir / 'heatmaps' / f'{output_name}_{prefix}_heatmap.png'
    overlay_path = output_dir / 'overlays' / f'{output_name}_{prefix}_overlay.png'

    save_gray_image(raw_path, saliency_map)
    heatmap = create_heatmap(saliency_map)
    cv2.imwrite(str(heatmap_path), heatmap[:, :, ::-1])
    overlay = create_overlay(original_image, saliency_map)
    cv2.imwrite(str(overlay_path), overlay[:, :, ::-1])
