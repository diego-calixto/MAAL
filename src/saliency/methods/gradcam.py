import warnings
import torch
import torch.nn.functional as F
from torch import nn

from src.saliency.utils import aggregate_segmentation_target, normalize_map


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cpu', target_type: str = 'predicted', threshold: float = 0.5):
        self.model = model
        self.device = device
        self.target_type = target_type
        self.threshold = threshold
        self.activations = None
        self.gradients = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: nn.Module) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # reset hook outputs for this call
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)
        if len(outputs) < 2:
            raise ValueError('Model must return classification and segmentation outputs.')

        y_seg = outputs[1]
        # heuristic: warn if y_seg looks like probabilities (in [0,1])
        try:
            if torch.all(y_seg >= 0.0) and torch.all(y_seg <= 1.0):
                warnings.warn('y_seg appears to be in [0,1]; it may be probabilities rather than logits. Grad-CAM works better on logits.')
        except Exception:
            pass

        target = aggregate_segmentation_target(y_seg, target_type=self.target_type, threshold=self.threshold)
        # ensure we have a scalar target for backward
        assert target.ndim == 0 or target.numel() == 1
        target.backward(retain_graph=False)

        # check that backward hook ran
        if self.gradients is None:
            raise RuntimeError('Backward hook was not triggered.')
        if self.activations is None or self.gradients is None:
            raise RuntimeError('Hooks did not capture activations or gradients. Check target layer selection.')

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        # If batch dim present, prefer returning single 2D map when batch==1.
        if cam.ndim == 3:
            if cam.shape[0] == 1:
                cam = cam[0]
            else:
                warnings.warn('GradCAM produced multiple maps (batch>1); returning the first map.')
                cam = cam[0]
        return normalize_map(cam)
