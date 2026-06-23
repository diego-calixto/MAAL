import re
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.utils.common import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, VALID_IMAGE_EXTENSIONS


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_checkpoint_structure(checkpoint: dict) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a dictionary.')
    if 'model_state' not in checkpoint:
        raise ValueError('Checkpoint missing "model_state" key.')


def list_image_files(image_dir: str) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {image_dir}')

    image_paths = []
    for path in sorted(image_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            image_paths.append(path)
    if not image_paths:
        raise ValueError(f'No image files found in {image_dir}')
    return image_paths


def load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(image: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def resize_map(saliency_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(saliency_map.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def normalize_map(saliency_map: np.ndarray) -> np.ndarray:
    saliency_map = saliency_map.astype(np.float32)
    min_val = float(np.nanmin(saliency_map))
    max_val = float(np.nanmax(saliency_map))
    if max_val - min_val < 1e-8:
        return np.zeros_like(saliency_map, dtype=np.float32)
    return np.clip((saliency_map - min_val) / (max_val - min_val), 0.0, 1.0)


def resolve_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    if layer_name:
        current = model
        tokens = layer_name.split('.')
        for token in tokens:
            list_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\[(-?\d+)\]', token)
            if list_match:
                attr, idx = list_match.group(1), int(list_match.group(2))
                current = getattr(current, attr)
                current = current[idx]
            else:
                current = getattr(current, token)
        if not isinstance(current, nn.Module):
            raise ValueError(f'Layer path {layer_name} did not resolve to a nn.Module.')
        return current

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer4'):
        layer4 = model.encoder.layer4
        if len(layer4) > 0 and hasattr(layer4[-1], 'conv3'):
            return layer4[-1].conv3
    raise ValueError('Unable to resolve a default target layer for Grad-CAM. Pass --target-layer explicitly.')


def aggregate_segmentation_target(y_seg: torch.Tensor, target_type: str = 'predicted', threshold: float = 0.5) -> torch.Tensor:
    if y_seg.ndim != 4:
        raise ValueError('Expected segmentation logits with shape [B, C, H, W].')

    if y_seg.shape[1] != 1:
        y_seg = y_seg[:, 0:1, :, :]

    if target_type == 'sum':
        return y_seg.sum()

    if target_type == 'mean':
        return y_seg.mean()

    if target_type == 'predicted':
        mask = torch.sigmoid(y_seg) > threshold
        if mask.sum() > 0:
            return (y_seg * mask.float()).sum()
        return y_seg.sum()

    if target_type == 'soft':
        probs = torch.sigmoid(y_seg)
        return (y_seg * probs).sum()

    raise ValueError(f'Unsupported target_type: {target_type}')


def make_model(model_type: str):
    model_type = model_type.lower()
    if model_type == 'fusion_cam':
        from src.models.fusion_cam_head import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward')
    if model_type == 'attention':
        from src.models.attention import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2)
    if model_type == 'cam_head':
        from src.models.cam_head import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2)
    if model_type == 'baseline':
        from src.models.baseline import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2)
    if model_type == 'maal':
        from src.models.maal import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward')
    if model_type == 'maal_expg':
        from src.models.maal_expG import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward')
    if model_type == 'maal_exph':
        from src.models.maal_expH import MultiTaskNetwork
        return MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward')
    raise ValueError(f'Unknown model_type: {model_type}. Available: fusion_cam, attention, cam_head, baseline, maal, maal_expg, maal_exph.')


def load_model_from_checkpoint(checkpoint_path: str, model_type: str, device: str) -> Tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    assert_checkpoint_structure(checkpoint)

    model = make_model(model_type)
    model.to(device)

    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError as err:
        err_msg = str(err)
        if 'fusion_conv.weight' in err_msg and 'fusion_weights' in checkpoint and hasattr(model, 'fusion_conv'):
            model.load_state_dict({k: v for k, v in checkpoint['model_state'].items() if k != 'fusion_conv.weight'}, strict=False)
            model.fusion_conv.weight.data.copy_(checkpoint['fusion_weights'].to(device))
        else:
            try:
                model.load_state_dict(checkpoint['model_state'], strict=False)
                warning_msg = (
                    f"Loaded checkpoint for model_type='{model_type}' with non-strict state_dict. "
                    f"Some layers may be missing or randomly initialized: {err_msg}"
                )
                print(f"WARNING: {warning_msg}")
            except RuntimeError:
                raise

    if hasattr(model, 'fusion_conv') and 'fusion_weights' in checkpoint:
        try:
            model.fusion_conv.weight.data.copy_(checkpoint['fusion_weights'].to(device))
        except Exception:
            pass

    model.eval()
    return model, checkpoint


def ensure_directories(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
