import argparse
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_cls=2, fusion_mode='learned_forward'):
        super(MultiTaskNetwork, self).__init__()
        if fusion_mode != 'learned_forward':
            raise ValueError("Only 'learned_forward' fusion_mode is supported.")
        self.encoder = SharedEncoder()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes_cls)

        self.saliency_stages = [1, 2, 3, 4]
        stage_channels = [256, 512, 1024, 2048]
        self.saliency_heads = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1, bias=False)
            for ch in stage_channels
        ])

        self.num_scales = len(self.saliency_heads)
        self.fusion_conv = nn.Conv2d(self.num_scales, 1, kernel_size=1, bias=False)

        self.decoder = SegmentationDecoder(num_classes=1)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1]

        x_cls_feat = self.avgpool(f_final)
        x_cls_flat = torch.flatten(x_cls_feat, 1)
        y_cls = self.fc(x_cls_flat)

        saliency_maps = []
        target_size = features[self.saliency_stages[0]].shape[2:]
        for idx, stage_idx in enumerate(self.saliency_stages):
            stage_feat = features[stage_idx]
            stage_map = self.saliency_heads[idx](stage_feat)
            stage_map = F.interpolate(
                stage_map,
                size=target_size,
                mode='bilinear',
                align_corners=True
            )
            saliency_maps.append(stage_map)

        saliency_stack = torch.cat(saliency_maps, dim=1)
        fused = self.fusion_conv(saliency_stack)
        y_seg = self.decoder(features)
        return y_cls, y_seg, fused, saliency_maps


class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_align = w_align
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, saliency_logits):
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)
        loss_seg = self.seg_criterion(y_seg_pred, y_seg_true)

        saliency_upsampled = F.interpolate(
            saliency_logits,
            size=y_seg_true.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        positive_mask = (y_cls_true == 1)

        if positive_mask.any():
            cam_pos = saliency_upsampled[positive_mask]
            mask_pos = y_seg_true[positive_mask]

            loss_align_bce = F.binary_cross_entropy_with_logits(cam_pos, mask_pos)

            with torch.cuda.amp.autocast(enabled=False):
                cam_fp32 = torch.sigmoid(cam_pos.float())
                mask_fp32 = mask_pos.float()

                intersection = (cam_fp32 * mask_fp32).sum(dim=(1, 2, 3))
                union = cam_fp32.sum(dim=(1, 2, 3)) + mask_fp32.sum(dim=(1, 2, 3))
                dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
                loss_align_dice = 1.0 - dice_score.mean()
        else:
            loss_align_bce = torch.tensor(0.0, device=y_seg_true.device)
            loss_align_dice = torch.tensor(0.0, device=y_seg_true.device)

        loss_align = loss_align_bce + loss_align_dice
        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_align * loss_align)
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


def visualize_model_predictions(model, loader, device, num_images=5, output_dir='validation_visualizations', fold=None):
    model.eval()
    images, targets_cls, targets_seg = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        y_cls, y_seg, saliency_logits, saliency_maps = model(images)
        preds_seg = (torch.sigmoid(y_seg) > 0.5).float()
        saliency_map = torch.sigmoid(saliency_logits)
        saliency_resized = F.interpolate(saliency_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True)
        B = saliency_resized.shape[0]
        s_min = saliency_resized.view(B, -1).min(1, keepdim=True)[0].view(B, 1, 1, 1)
        s_max = saliency_resized.view(B, -1).max(1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency_resized = (saliency_resized - s_min) / (s_max - s_min + 1e-8)

        stage_positions = [
            (0, 2),  # Stage1
            (0, 3),  # Stage2
            (1, 0),  # Stage3
            (1, 1),  # Stage4
        ]

        if fold is not None:
            cams_out_dir = os.path.join(output_dir, f'fold_{fold}')
            filename_prefix = f'fold_{fold}_'
        else:
            cams_out_dir = output_dir
            filename_prefix = ''

        os.makedirs(cams_out_dir, exist_ok=True)
        for i in range(min(num_images, images.size(0))):
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            img_show = unnormalize_image(images[i])
            fused_np = saliency_resized[i].cpu().squeeze().numpy()

            axs[0, 0].imshow(img_show)
            cls_pred_idx = torch.argmax(y_cls[i]).item()
            label_text = "Positivo" if cls_pred_idx == 1 else "Negativo"
            color = 'green' if cls_pred_idx == targets_cls[i].item() else 'red'
            axs[0, 0].set_title(f"Original\nPred: {label_text}", color=color)
            axs[0, 0].axis('off')

            axs[0, 1].imshow(targets_seg[i].cpu().squeeze(), cmap='gray')
            axs[0, 1].set_title("GT")
            axs[0, 1].axis('off')

            for j, stage_map in enumerate(saliency_maps):
                stage_up = F.interpolate(stage_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True)
                stage_prob = torch.sigmoid(stage_up)[i]
                smin = stage_prob.view(-1).min()
                smax = stage_prob.view(-1).max()
                stage_norm = (stage_prob - smin) / (smax - smin + 1e-8)
                row, col = stage_positions[j]
                axs[row, col].imshow(stage_norm.cpu().squeeze(), cmap='jet')
                axs[row, col].set_title(f"Stage {j+1}")
                axs[row, col].axis('off')

            axs[1, 2].imshow(fused_np, cmap='jet')
            axs[1, 2].set_title("Fused")
            axs[1, 2].axis('off')

            axs[1, 3].imshow(preds_seg[i].cpu().squeeze(), cmap='gray')
            axs[1, 3].set_title("Segmentation")
            axs[1, 3].axis('off')

            plt.tight_layout()
            fig_path = os.path.join(cams_out_dir, f'{filename_prefix}image_{i}_combined_cam_summary.png')
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Saved combined CAM figure: {fig_path}")

    print(f"Saved validation visualizations to: {cams_out_dir}")


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

            # After training, log learned fusion weights for each fold if available
            for fold in range(N_SPLITS):
                best_path = os.path.join(args.checkpoint_dir, 'MAAL', f'fold_{fold}', 'best.pt')
                if os.path.exists(best_path):
                    ckpt = torch.load(best_path, map_location=DEVICE)
                    model = MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward').to(DEVICE)
                    model.load_state_dict(ckpt['model_state'])
                    try:
                        weights = model.fusion_conv.weight.data.squeeze().cpu().numpy()
                        print(f"Fold {fold} fusion weights: {weights}")
                    except Exception:
                        print(f"Fold {fold} fusion weights: unable to read weights")
                else:
                    print(f"No best checkpoint for fold {fold} at {best_path}")
        else:
            print("DataFrame vazio. Verifique se o dataset foi gerado corretamente.")
    else:
        print(f"Pasta {DATASET_DIR} não encontrada. Rode o script de 'create_balanced_dataset' primeiro.")


if __name__ == '__main__':
    main()
