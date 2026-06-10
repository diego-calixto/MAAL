import argparse
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from common import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
    DATASET_DIR,
    N_SPLITS,
    USE_COMPILE,
    ALIGNMENT_WEIGHT,
    SharedEncoder,
    SegmentationDecoder,
    CrackDataset,
    prepare_dataframe,
    run_training_pipeline,
)


class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_cls=2, fusion_mode='learned_forward'):
        super(MultiTaskNetwork, self).__init__()
        self.encoder = SharedEncoder()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes_cls)

        self.saliency_stages = [1, 2, 3, 4]
        stage_channels = [256, 512, 1024, 2048]
        self.saliency_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
            for ch in stage_channels
        ])

        self.fusion_mode = fusion_mode
        self.num_scales = len(self.saliency_heads)
        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        self.fusion_conv = nn.Conv2d(self.num_scales, 1, kernel_size=1, bias=False)

        self.decoder = SegmentationDecoder(num_classes=1)
        self._validate_fusion_mode()

    def _validate_fusion_mode(self):
        valid_modes = {'mean', 'fixed_weighted', 'best_layer', 'learned_forward'}
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"Fusion mode '{self.fusion_mode}' is inválido. Use um dos: {valid_modes}.")

    def _normalize_map(self, saliency_map):
        batch = saliency_map.size(0)
        flat = saliency_map.view(batch, -1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(batch, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(batch, 1, 1, 1)
        return (saliency_map - min_val) / (max_val - min_val + 1e-8)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1]

        x_cls_feat = self.avgpool(f_final)
        x_cls_flat = torch.flatten(x_cls_feat, 1)
        y_cls = self.fc(x_cls_flat)

        saliency_maps = []
        target_size = f_final.shape[2:]
        for idx, stage_idx in enumerate(self.saliency_stages):
            stage_feat = features[stage_idx]
            stage_map = self.saliency_heads[idx](stage_feat)
            stage_map = F.interpolate(stage_map, size=target_size, mode='bilinear', align_corners=True)
            stage_map = self._normalize_map(stage_map)
            saliency_maps.append(stage_map)

        saliency_stack = torch.cat(saliency_maps, dim=1)
        if self.fusion_mode == 'mean':
            fused = saliency_stack.mean(dim=1, keepdim=True)
        elif self.fusion_mode == 'fixed_weighted':
            weights = torch.softmax(self.fusion_weights, dim=0).view(1, self.num_scales, 1, 1)
            fused = (saliency_stack * weights).sum(dim=1, keepdim=True)
        elif self.fusion_mode == 'best_layer':
            scores = saliency_stack.view(saliency_stack.size(0), self.num_scales, -1).mean(dim=2)
            best_idx = scores.argmax(dim=1)
            one_hot = F.one_hot(best_idx, num_classes=self.num_scales).float().view(
                saliency_stack.size(0), self.num_scales, 1, 1
            )
            fused = (saliency_stack * one_hot).sum(dim=1, keepdim=True)
        else:
            fused = self.fusion_conv(saliency_stack)
            fused = F.relu(fused)

        saliency_map = self._normalize_map(fused)
        y_seg = self.decoder(features)
        return y_cls, y_seg, saliency_map


class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_align = w_align
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, saliency_map):
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)
        loss_seg = self.seg_criterion(y_seg_pred, y_seg_true)

        saliency_upsampled = F.interpolate(saliency_map, size=y_seg_true.shape[2:], mode='bilinear', align_corners=True)
        background_mask = 1.0 - y_seg_true
        loss_align = torch.mean(torch.abs(saliency_upsampled * background_mask))

        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_align * loss_align)
        return total_loss, {"cls": loss_cls, "seg": loss_seg, "align": loss_align}


def unnormalize_image(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def visualize_model_predictions(model, loader, device, num_images=5):
    model.eval()
    images, targets_cls, targets_seg = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        y_cls, y_seg, saliency_map = model(images)
        preds_seg = (torch.sigmoid(y_seg) > 0.5).float()
        saliency_resized = F.interpolate(saliency_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True)
        B = saliency_resized.shape[0]
        s_min = saliency_resized.view(B, -1).min(1, keepdim=True)[0].view(B, 1, 1, 1)
        s_max = saliency_resized.view(B, -1).max(1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency_resized = (saliency_resized - s_min) / (s_max - s_min + 1e-8)

    plt.figure(figsize=(16, 4 * num_images))
    for i in range(min(num_images, images.size(0))):
        plt.subplot(num_images, 4, i * 4 + 1)
        img_show = unnormalize_image(images[i])
        plt.imshow(img_show)
        cls_pred_idx = torch.argmax(y_cls[i]).item()
        label_text = "Positivo" if cls_pred_idx == 1 else "Negativo"
        color = 'green' if cls_pred_idx == targets_cls[i].item() else 'red'
        plt.title(f"Img Original\nPred: {label_text}", color=color)
        plt.axis('off')

        plt.subplot(num_images, 4, i * 4 + 2)
        plt.imshow(targets_seg[i].cpu().squeeze(), cmap='gray')
        plt.title("Ground Truth (Mask)")
        plt.axis('off')

        plt.subplot(num_images, 4, i * 4 + 3)
        plt.imshow(preds_seg[i].cpu().squeeze(), cmap='gray')
        plt.title("Segmentação Predita")
        plt.axis('off')

        plt.subplot(num_images, 4, i * 4 + 4)
        plt.imshow(img_show)
        plt.imshow(saliency_resized[i].cpu().squeeze(), cmap='jet', alpha=0.5)
        plt.title("Saliency Map (CAM)\n(Onde a rede olhou)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train MAAL multi-task model with checkpoint support.')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoint files')
    parser.add_argument('--resume-from', default=None, help='Path to a checkpoint file to resume training from')
    args = parser.parse_args()

    if os.path.exists(DATASET_DIR):
        df = prepare_dataframe(DATASET_DIR)
        print(f"DataFrame carregado com {len(df)} imagens.")

        if len(df) > 0:
            run_training_pipeline(
                run_name='MAAL',
                model_factory=lambda: MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward'),
                criterion_factory=lambda: MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT),
                df=df,
                n_splits=N_SPLITS,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from
            )
        else:
            print("DataFrame vazio. Verifique se o dataset foi gerado corretamente.")
    else:
        print(f"Pasta {DATASET_DIR} não encontrada. Rode o script de 'create_balanced_dataset' primeiro.")


if __name__ == '__main__':
    main()
