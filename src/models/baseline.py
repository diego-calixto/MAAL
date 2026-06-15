import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ..utils.common import (
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
    def __init__(self, num_classes_cls=2):
        super(MultiTaskNetwork, self).__init__()
        self.encoder = SharedEncoder()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes_cls)
        self.decoder = SegmentationDecoder(num_classes=1)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1]

        x_cls_feat = self.avgpool(f_final)
        x_cls_flat = torch.flatten(x_cls_feat, 1)
        y_cls = self.fc(x_cls_flat)

        y_seg = self.decoder(features)
        return y_cls, y_seg


class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_align=0.0):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        # alignment weight retained for API compatibility but ignored for baseline
        self.w_align = w_align
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()
        self.current_epoch = 0
        self.batch_count = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.batch_count = 0

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, saliency_logits=None):
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)
        loss_seg = self.seg_criterion(y_seg_pred, y_seg_true)

        # For baseline we do not compute alignment losses; return zeros for compatibility
        loss_align_bce = torch.tensor(0.0, device=y_seg_true.device)
        loss_align_dice = torch.tensor(0.0, device=y_seg_true.device)
        loss_align = torch.tensor(0.0, device=y_seg_true.device)

        # Sanity check: print loss components in first epoch, first 3 batches
        self.batch_count += 1
        if self.current_epoch == 0 and self.batch_count <= 3:
            print(f"Batch {self.batch_count}: cls={loss_cls.item():.4f}, seg={loss_seg.item():.4f}")

        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg)
        return total_loss, {
            "cls": loss_cls.detach(),
            "seg": loss_seg.detach(),
            "align_bce": loss_align_bce.detach(),
            "align_dice": loss_align_dice.detach(),
            "align": loss_align.detach()
        }


def unnormalize_image(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def visualize_model_predictions(model, loader, device, num_images=5, output_dir='validation_visualizations'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    images, targets_cls, targets_seg = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            y_cls, y_seg = outputs
            saliency_resized = torch.zeros((images.size(0), 1, IMAGE_SIZE, IMAGE_SIZE), device=images.device)
        else:
            # fallback for models that return saliency logits
            y_cls, y_seg, saliency_logits = outputs
            preds_seg = (torch.sigmoid(y_seg) > 0.5).float()
            saliency_map = torch.sigmoid(saliency_logits)
            saliency_resized = F.interpolate(saliency_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True)
            B = saliency_resized.shape[0]
            s_min = saliency_resized.view(B, -1).min(1, keepdim=True)[0].view(B, 1, 1, 1)
            s_max = saliency_resized.view(B, -1).max(1, keepdim=True)[0].view(B, 1, 1, 1)
            saliency_resized = (saliency_resized - s_min) / (s_max - s_min + 1e-8)

        preds_seg = (torch.sigmoid(y_seg) > 0.5).float()

    for i in range(min(num_images, images.size(0))):
        img_show = unnormalize_image(images[i])
        sample_prefix = f"sample_{i+1}"

        plt.imsave(os.path.join(output_dir, f"{sample_prefix}_image.png"), img_show)
        plt.imsave(os.path.join(output_dir, f"{sample_prefix}_gt_mask.png"), targets_seg[i].cpu().squeeze(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"{sample_prefix}_pred_mask.png"), preds_seg[i].cpu().squeeze(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"{sample_prefix}_cam_heatmap.png"), saliency_resized[i].cpu().squeeze(), cmap='jet')

        overlay = img_show.copy()
        heatmap = plt.get_cmap('jet')(saliency_resized[i].cpu().squeeze())[..., :3]
        overlay = np.clip(overlay * 0.5 + heatmap * 0.5, 0, 1)
        plt.imsave(os.path.join(output_dir, f"{sample_prefix}_cam_overlay.png"), overlay)

    print(f"Saved validation visualizations to: {output_dir}")


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
                model_factory=lambda: MultiTaskNetwork(num_classes_cls=2),
                criterion_factory=lambda: MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=0.0),
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
