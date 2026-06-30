import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.saliency.config import (
    AVAILABLE_METHODS,
    AVAILABLE_MODELS,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METHOD,
    DEFAULT_MODEL_TYPE,
    DEFAULT_TARGET_LAYER,
    DEFAULT_THRESHOLD,
    OUTPUT_DIR,
)
from src.saliency.methods import GradCAM, VanillaSaliency
from src.saliency.utils import (
    assert_checkpoint_structure,
    ensure_directories,
    list_image_files,
    load_rgb_image,
    load_model_from_checkpoint,
    preprocess_image,
    resize_map,
    resolve_target_layer,
    set_seed,
)
from src.saliency.visualization import save_saliency_outputs, create_heatmap, create_overlay


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate saliency maps for segmentation models.')
    parser.add_argument('--checkpoint', required=True, help='Path to trained checkpoint file (best.pt).')
    parser.add_argument('--input-dir', required=True, help='Directory with input images to explain.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Base directory for saved explanation outputs.')
    parser.add_argument('--model-type', default=DEFAULT_MODEL_TYPE, choices=AVAILABLE_MODELS, help='Model architecture used for training.')
    parser.add_argument('--method', default=DEFAULT_METHOD, choices=AVAILABLE_METHODS, help='Saliency method to compute.')
    parser.add_argument('--target-layer', default=DEFAULT_TARGET_LAYER, help='Target layer for Grad-CAM. Use full module path if needed.')
    parser.add_argument('--target-type', default='predicted', choices=['predicted', 'sum', 'mean', 'soft'], help='How to aggregate segmentation logits into a scalar target.')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Input size used when preprocessing images.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Threshold used for segmentation target mask creation.')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='Device to run inference on: cpu or cuda.')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process (for quick examples).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # When using the FusionCAM head we ignore the saliency ``method`` argument because
    # the model already produces a fused saliency map.
    parser.add_argument('--fusion-cam', action='store_true', help='If set, use the fused saliency map from the FusionCAM model (model_type "fusion_cam_head").')
    return parser.parse_args()


def build_explainer(method: str, model: torch.nn.Module, device: str, target_type: str, threshold: float, target_layer_name: str):
    """Create a saliency explainer unless using FusionCAM model.

    For the FusionCAM model (identified by the model_type 'fusion_cam_head'),
    the saliency map is produced directly by the model's forward pass and does
    not require a separate explainer such as GradCAM or VanillaSaliency.
    """
    # The FusionCAM model returns a fused saliency map as the third output of its forward method.
    # The caller will handle this case, so we raise an informative error if an explainer is
    # requested for this architecture.
    if method == 'fusion_cam_head':
        raise ValueError('Explainer not required for FusionCAM model; use model output directly.')
    if method == 'vanilla':
        return VanillaSaliency(model=model, device=device, target_type=target_type, threshold=threshold)
    if method == 'gradcam':
        target_layer = resolve_target_layer(model, layer_name=target_layer_name)
        return GradCAM(model=model, target_layer=target_layer, device=device, target_type=target_type, threshold=threshold)
    raise ValueError(f'Unsupported method: {method}')


def process_image(
    image_path: Path,
    model: torch.nn.Module,
    device: str,
    use_fused: bool,
    explainer,
    image_size: int,
    output_dir: Path,
) -> None:
    """Generate and save a saliency map for a single image, plus a combined figure.

    If ``use_fused`` is True (i.e., the model is a FusionCAM head), the saliency
    map is obtained from the model's forward output. Otherwise, an explainer
    object (GradCAM or VanillaSaliency) is used.
    """
    original_image = load_rgb_image(image_path)
    input_tensor = preprocess_image(original_image, image_size=image_size)
    input_tensor = input_tensor.to(device)
    if use_fused:
        # Model expects a batch dimension.
        with torch.no_grad():
            y_cls, y_seg, fused, _ = model(input_tensor.unsqueeze(0))
        # fused saliency map
        saliency_map = torch.sigmoid(fused.squeeze(0))
        # Predicted segmentation mask (binary)
        seg_prob = torch.sigmoid(y_seg.squeeze(0))
        pred_mask = (seg_prob > 0.5).float()
    else:
        saliency_map = explainer.generate(input_tensor)
        pred_mask = None
    saliency_map_original = resize_map(saliency_map, original_image.shape[:2])
    prefix = 'fusion_cam' if use_fused else explainer.__class__.__name__.lower()
    # Save individual saliency outputs (raw, heatmap, overlay)
    save_saliency_outputs(image_path, original_image, saliency_map_original, output_dir, prefix=prefix)
    # Create combined visualization figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Original image
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')
    # Predicted mask if available
    if pred_mask is not None:
        axs[0, 1].imshow(pred_mask.cpu().numpy(), cmap='gray')
        axs[0, 1].set_title('Predicted Mask')
        axs[0, 1].axis('off')
    else:
        axs[0, 1].axis('off')
    # Heatmap
    heatmap_img = create_heatmap(saliency_map_original)
    axs[1, 0].imshow(heatmap_img)
    axs[1, 0].set_title('Heatmap')
    axs[1, 0].axis('off')
    # Overlay
    overlay_img = create_overlay(original_image, saliency_map_original)
    axs[1, 1].imshow(overlay_img)
    axs[1, 1].set_title('Overlay')
    axs[1, 1].axis('off')
    combined_path = output_dir / f"{image_path.stem}_{prefix}_combined.png"
    fig.tight_layout()
    fig.savefig(combined_path, bbox_inches='tight')
    plt.close(fig)




def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    output_dir = Path(args.output_dir)
    ensure_directories(output_dir / 'raw', output_dir / 'heatmaps', output_dir / 'overlays')

    model, checkpoint = load_model_from_checkpoint(str(checkpoint_path), args.model_type, device)
    assert_checkpoint_structure(checkpoint)

    image_paths = list_image_files(args.input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    # Determine whether we should use the model's fused saliency map.
    use_fused = args.model_type == 'fusion_cam_head'
    explainer = None
    if not use_fused:
        # For standard models we fall back to the requested saliency method.
        explainer = build_explainer(args.method, model, device, args.target_type, args.threshold, args.target_layer)

    print(f'Loaded {len(image_paths)} images from {args.input_dir}')
    print(f'Using model_type={args.model_type}, method={args.method}, device={device}')
    if use_fused:
        print('Using FusionCAM fused saliency map directly from model output.')
    else:
        print(f'Using explainer: {explainer.__class__.__name__}')

    for image_path in tqdm(image_paths, desc='Generating saliency maps'):
        try:
            process_image(
                image_path,
                model,
                device,
                use_fused,
                explainer,
                args.image_size,
                output_dir,
            )
        except Exception as exc:
            print(f'ERROR: Failed to process {image_path}: {exc}')

    print(f'Saliency generation complete. Saved outputs to {output_dir}')


if __name__ == '__main__':
    main()
