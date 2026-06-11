"""Multi-task crack detection with learned attention supervision.

Combines classification, segmentation, and spatial attention regularization for robust crack evidence learning.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.common import (
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

ATTENTION_FEATURE_LEVEL = 3


class AttentionHead(nn.Module):
    def __init__(self, in_channels=2048):
        super(AttentionHead, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.attention(x)


class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_cls=2):
        super(MultiTaskNetwork, self).__init__()
        self.encoder = SharedEncoder()

        attention_feature_channels = {
            2: 512,
            3: 1024,
            4: 2048
        }.get(ATTENTION_FEATURE_LEVEL, 2048)
        self.attention_head = AttentionHead(in_channels=attention_feature_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes_cls)
        self.decoder = SegmentationDecoder(num_classes=1)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1]

        attention_features = features[ATTENTION_FEATURE_LEVEL]
        attention_logits = self.attention_head(attention_features)
        attention_map = torch.sigmoid(attention_logits)
        attention_map_resized = F.interpolate(
            attention_map,
            size=f_final.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        gated_features = f_final * attention_map_resized
        x_cls_feat = self.avgpool(gated_features)
        x_cls_flat = torch.flatten(x_cls_feat, 1)
        y_cls = self.fc(x_cls_flat)

        y_seg = self.decoder(features)
        return y_cls, y_seg, attention_logits, attention_map


class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_attn=ALIGNMENT_WEIGHT):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_attn = w_attn
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()
        self.attn_criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, attention_logits):
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)

        y_seg_target = F.interpolate(
            y_seg_true,
            size=y_seg_pred.shape[2:],
            mode='bilinear',
            align_corners=True
        ).float()

        loss_seg = self.seg_criterion(y_seg_pred, y_seg_target)

        attention_target = F.interpolate(
            y_seg_target,
            size=attention_logits.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        attention_loss_bce = self.attn_criterion(attention_logits, attention_target)

        attention_map = torch.sigmoid(attention_logits)
        attn_intersection = (attention_map * attention_target).sum(dim=(1, 2, 3))
        attn_union = attention_map.sum(dim=(1, 2, 3)) + attention_target.sum(dim=(1, 2, 3))
        attn_dice = 1.0 - (2.0 * attn_intersection + 1e-8) / (attn_union + 1e-8)
        attn_dice = attn_dice.mean()

        loss_attention = attention_loss_bce + attn_dice
        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_attn * loss_attention)

        return total_loss, {
            'cls': loss_cls.detach(),
            'seg': loss_seg.detach(),
            'attention_bce': attention_loss_bce.detach(),
            'attention_dice': attn_dice.detach()
        }


def unnormalize_image(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def visualize_predictions(model, loader, device, out_dir='visuals', n_samples=16, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for images, targets_cls, targets_seg in loader:
            images = images.to(device)
            targets_seg = targets_seg.to(device)

            logits, y_seg, attention_logits, attention_map = model(images)
            probs = torch.sigmoid(y_seg)
            preds_mask = (probs > threshold).float()

            for i in range(images.shape[0]):
                if saved >= n_samples:
                    return

                img_t = images[i].cpu()
                img_np = img_t.clone()
                for ch in range(3):
                    img_np[ch] = img_np[ch] * IMAGENET_STD[ch] + IMAGENET_MEAN[ch]
                img_np = (img_np.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

                gt = targets_seg[i].cpu().squeeze(0).numpy()
                pred = preds_mask[i].cpu().squeeze(0).numpy()

                attn = F.interpolate(attention_map[i:i+1], size=(img_np.shape[0], img_np.shape[1]), mode='bilinear', align_corners=True)
                attn_np = attn.squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_np)
                axes[0].set_title('Input')
                axes[0].axis('off')

                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('GT Mask')
                axes[1].axis('off')

                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Pred Mask')
                axes[2].axis('off')

                hm = plt.cm.get_cmap('hot')(attn_np)[:, :, :3]
                overlay = (0.6 * img_np.astype(np.float32) / 255.0) + (0.4 * hm)
                overlay = np.clip(overlay, 0, 1)
                axes[3].imshow(overlay)
                axes[3].set_title('Attention Overlay')
                axes[3].axis('off')

                out_path = os.path.join(out_dir, f'vis_{saved:03d}.png')
                plt.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                saved += 1


def main():
    if os.path.exists(DATASET_DIR):
        df = prepare_dataframe(DATASET_DIR)
        print(f"DataFrame loaded with {len(df)} images.")

        if len(df) > 0:
            run_training_pipeline(
                run_name='Attention',
                model_factory=lambda: MultiTaskNetwork(num_classes_cls=2),
                criterion_factory=lambda: MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_attn=ALIGNMENT_WEIGHT),
                df=df,
                n_splits=N_SPLITS
            )
        else:
            print("DataFrame empty. Check if dataset was generated correctly.")
    else:
        print(f"Dataset directory {DATASET_DIR} not found. Run dataset creation script first.")


if __name__ == '__main__':
    main()
