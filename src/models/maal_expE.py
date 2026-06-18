import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1].parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    prepare_dataframe,
)
from src.utils.train_maal import run_maal_training_pipeline


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

    #MAAL
    def __init__(self, w_cls=1.0, w_seg=1.0, w_align=1.0, num_scales=4):
        """
        w_cls: Peso da perda de classificação
        w_seg: Peso da perda de segmentação
        w_align: Peso total do módulo MAAL
        num_scales: Número de camadas extraídas no encoder (padrão 4 para ResNet)
        """
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_align = w_align
        
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()
        
        # Ponderador Adaptativo MAAL (alpha_l)
        # Parâmetros aprendíveis inicializados com 1.0
        self.alpha_weights = nn.Parameter(
            torch.tensor([-1.0, -0.5, 0.5, 1.0])
        )
        self.alpha_temperature = 2.0
        
        self.current_epoch = 0
        self.batch_count = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.batch_count = 0 

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, saliency_maps):
        # 1. Perdas Padrão das Tarefas
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)
        loss_seg = self.seg_criterion(y_seg_pred, y_seg_true)
        
        # Inicializa a perda MAAL
        loss_maal = torch.tensor(0.0, device=y_seg_true.device)
        loss_align_bce_total = torch.tensor(0.0, device=y_seg_true.device)
        loss_align_dice_total = torch.tensor(0.0, device=y_seg_true.device)
        
        # 2. Calcula os pesos adaptativos (Softmax garante que a soma de alpha_l = 1)
        alphas = F.softmax(
            self.alpha_weights * self.alpha_temperature,
            dim=0
        )

        target_alpha = torch.tensor(
            [0.05, 0.15, 0.30, 0.50],
            device=alphas.device
        )
        loss_alpha_reg = F.mse_loss(alphas, target_alpha)
        
        # 3. Cálculo do Alinhamento MAAL
        # O alinhamento só faz sentido se a imagem contiver o defeito (classe positiva = 1)
        cls_confidence = torch.softmax(
            y_cls_pred.detach(),
            dim=1
        )[:, 1]

        positive_mask = (
            (y_cls_true == 1)
            &
            (cls_confidence > 0.70)
        )
        
        if positive_mask.any():
            # Filtra apenas os Ground Truths das imagens com trincas
            target_mask = y_seg_true[positive_mask] 
            
            # Itera sobre cada nível hierárquico (l)
            for l, s_map in enumerate(saliency_maps):
                # Faz o upsample do mapa de saliência para o tamanho original da imagem
                s_map_up = F.interpolate(
                    s_map, 
                    size=y_seg_true.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
                
                # Filtra as imagens positivas para o batch atual
                cam_pos = s_map_up[positive_mask]
                
                # Perda de Alinhamento Local (L_align^(l)) - BCE
                # Usa BCEWithLogits para empurrar as ativações da saliência em direção à máscara real
                loss_align_l_bce = F.binary_cross_entropy_with_logits(cam_pos, target_mask)
                
                # Perda de Alinhamento Local (L_align^(l)) - Dice
                # Calcula Dice em precisão FP32 para maior estabilidade numérica
                with torch.cuda.amp.autocast(enabled=False):
                    cam_fp32 = torch.sigmoid(cam_pos.float())
                    mask_fp32 = target_mask.float()
                    
                    intersection = (cam_fp32 * mask_fp32).sum(dim=(1, 2, 3))
                    union = cam_fp32.sum(dim=(1, 2, 3)) + mask_fp32.sum(dim=(1, 2, 3))
                    dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
                    loss_align_l_dice = 1.0 - dice_score.mean()
                
                # Combina BCE + Dice
                loss_align_l = loss_align_l_bce + loss_align_l_dice
                
                # Soma Ponderada Adaptativa (alpha_l * L_align^(l))
                stage_factor = [0.25, 0.50, 1.0, 1.5][l]
                loss_maal += alphas[l] * stage_factor * loss_align_l
                loss_align_bce_total = loss_align_bce_total + loss_align_l_bce.detach()
                loss_align_dice_total = loss_align_dice_total + loss_align_l_dice.detach()

        # 4. Perda Total do Modelo
        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_align * loss_maal) + (0.05 * loss_alpha_reg)
        
        # Diagnóstico: imprime componentes de perda nos primeiros 3 batches do primeiro epoch
        self.batch_count += 1
        if self.current_epoch == 0 and self.batch_count <= 3:
            print(f"Batch {self.batch_count}: cls={loss_cls.item():.4f}, seg={loss_seg.item():.4f}, "
                  f"align_bce={loss_align_bce_total.item():.4f}, align_dice={loss_align_dice_total.item():.4f}")
        
        return total_loss, {
            "cls": loss_cls.detach(),
            "seg": loss_seg.detach(),
            "align_bce": loss_align_bce_total.detach(),
            "align_dice": loss_align_dice_total.detach(),
            "align_maal": loss_maal.detach(),
            "alphas": alphas.detach() # Permite monitorar qual camada a rede julga mais importante
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
            run_maal_training_pipeline(
                model_factory=lambda: MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward'),
                criterion_factory=lambda: MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT, num_scales=4),
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
