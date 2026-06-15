import torch

from src.saliency.utils import aggregate_segmentation_target, normalize_map


class VanillaSaliency:
    def __init__(self, model, device: str = 'cpu', target_type: str = 'predicted', threshold: float = 0.5):
        self.model = model
        self.device = device
        self.target_type = target_type
        self.threshold = threshold

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        # ensure gradients are cleared
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor)
        if len(outputs) < 2:
            raise ValueError('Model must return classification and segmentation outputs.')

        y_seg = outputs[1]
        # heuristic: warn if y_seg looks like probabilities
        try:
            if torch.all(y_seg >= 0.0) and torch.all(y_seg <= 1.0):
                import warnings

                warnings.warn('y_seg appears to be in [0,1]; it may be probabilities rather than logits. Vanilla saliency works better on logits.')
        except Exception:
            pass

        target = aggregate_segmentation_target(y_seg, target_type=self.target_type, threshold=self.threshold)
        # ensure scalar target
        assert target.ndim == 0 or target.numel() == 1
        target.backward(retain_graph=False)

        gradients = input_tensor.grad
        if gradients is None:
            raise RuntimeError('Input gradients are None; backward may not have been triggered.')

        saliency = gradients.abs().amax(dim=1)[0].cpu().numpy()
        return normalize_map(saliency)
