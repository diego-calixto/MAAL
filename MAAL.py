import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# =============================================================================
# 1. ARQUITETURA DO MODELO (AJUSTADA PARA CAM/SALIENCY CORRETO)
# =============================================================================

class SharedEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(SharedEncoder, self).__init__()
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(pretrained=pretrained)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []
        x0 = self.initial(x)
        features.append(x0)
        x1 = self.layer1(x0)
        features.append(x1)
        x2 = self.layer2(x1)
        features.append(x2)
        x3 = self.layer3(x2)
        features.append(x3)
        x4 = self.layer4(x3)
        features.append(x4)
        return features

class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(SegmentationDecoder, self).__init__()
        # Decoder estilo U-Net simplificado
        self.up4 = self.conv_block(2048 + 1024, 1024)
        self.up3 = self.conv_block(1024 + 512, 512)
        self.up2 = self.conv_block(512 + 256, 256)

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def match_tensor(self, tensor_target, tensor_ref):
        if tensor_target.size()[2:] != tensor_ref.size()[2:]:
            return F.interpolate(tensor_target, size=tensor_ref.size()[2:], mode='bilinear', align_corners=True)
        return tensor_target

    def forward(self, features):
        # Ajuste para usar as features corretas para as skip connections e o caminho principal
        # features: [x0, x1, x2, x3, x4]
        # x0 é a saída do initial (após maxpool), x1 é layer1, ..., x4 é layer4

        # Features que precisamos:
        # features[4] = x4 (final do encoder)
        # features[3] = x3
        # features[2] = x2
        # features[1] = x1

        d4 = self.up4(torch.cat([features[4], self.match_tensor(features[3], features[4])], dim=1))
        d3 = self.up3(torch.cat([d4, self.match_tensor(features[2], d4)], dim=1))
        d2 = self.up2(torch.cat([d3, self.match_tensor(features[1], d3)], dim=1))
        out_seg = self.final_conv(d2)
        return out_seg

class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_cls=2, fusion_mode='learned_forward'):
        super(MultiTaskNetwork, self).__init__()
        self.encoder = SharedEncoder()

        # Branch de Classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Note: Bias=False é importante para o CAM puro, mas com Bias funciona se tratarmos corretamente.
        self.fc = nn.Linear(2048, num_classes_cls)
        # Branch de Saliência Multi-Escala
        # Usamos mapas de diferentes profundidades para capturar características de baixo e alto nível.
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
        BATCH = saliency_map.size(0)
        flat = saliency_map.view(BATCH, -1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(BATCH, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(BATCH, 1, 1, 1)
        return (saliency_map - min_val) / (max_val - min_val + 1e-8)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1] # Shape: (Batch, 2048, H, W) -> ex: (Batch, 2048, 7, 7)

        # 1. Branch Classificação
        x_cls_feat = self.avgpool(f_final)
        x_cls_flat = torch.flatten(x_cls_feat, 1)
        y_cls = self.fc(x_cls_flat)

        # 2. Branch Saliência Multi-Escala
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
        elif self.fusion_mode == 'learned_forward':
            fused = self.fusion_conv(saliency_stack)
            fused = F.relu(fused)
        else:
            raise ValueError(f"Modo de fusão desconhecido: {self.fusion_mode}")

        saliency_map = self._normalize_map(fused)

        # 3. Branch Segmentação
        y_seg = self.decoder(features)

        return y_cls, y_seg, saliency_map

# =============================================================================
# 2. FUNÇÃO DE LOSS (MultiTaskLoss) - Mantida a lógica, mas agora recebe CAM real
# =============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_seg=1.0, w_align=0.5):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_align = w_align
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, saliency_map):
        # 1. Loss de Classificação
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)

        # 2. Loss de Segmentação
        loss_seg = self.seg_criterion(y_seg_pred, y_seg_true)

        # 3. Loss de Alinhamento
        # O objetivo é: Onde a segmentação diz que é FUNDO (0), a saliência deve ser BAIXA.

        # Upsample do mapa de saliência (que vem 7x7) para o tamanho da máscara (224x224)
        saliency_upsampled = F.interpolate(saliency_map, size=y_seg_true.shape[2:], mode='bilinear', align_corners=True)

        # Inverter a máscara de Ground Truth: Onde é 1 vira 0, onde é 0 vira 1.
        # Queremos penalizar ativações no BACKGROUND.
        background_mask = 1.0 - y_seg_true

        # A Loss é a magnitude da Saliência nas áreas de Background
        alignment_term = saliency_upsampled * background_mask
        loss_align = torch.mean(torch.abs(alignment_term))

        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_align * loss_align)

        return total_loss, {"cls": loss_cls, "seg": loss_seg, "align": loss_align}

# =============================================================================
# 3. DATASET E PREPARAÇÃO (Versão Atualizada com Máscaras PNG)
# =============================================================================

class CrackDataset(Dataset):
    def __init__(self, df, img_size=224, augment=False):
        self.df = df
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Carregar Imagem
        image = cv2.imread(row['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carregar Máscara
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            # Fallback seguro: máscara zerada
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # Resize fixo para imagem e máscara, garantindo que os dois estejam alinhados.
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image = self.to_tensor(image)
        image = self.normalize(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, torch.tensor(row['label'], dtype=torch.long), mask


def prepare_dataframe(dataset_dir):
    data = []
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for label_name in ['Positive', 'Negative']:
        class_dir_img = os.path.join(dataset_dir, label_name, 'images')
        if not os.path.exists(class_dir_img):
            continue

        label = 1 if label_name == 'Positive' else 0
        mask_dir = os.path.join(dataset_dir, label_name, 'masks')

        for entry in os.scandir(class_dir_img):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in valid_exts:
                continue

            group_id = entry.name.split('_')[0]
            mask_name = os.path.splitext(entry.name)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"WARNING: Máscara não encontrada para {entry.path}. Usando máscara zerada.")

            data.append({
                'path': entry.path,
                'mask_path': mask_path,
                'filename': entry.name,
                'label': label,
                'group': group_id
            })
    return pd.DataFrame(data)

# =============================================================================
# 4. FUNÇÕES DE TREINAMENTO E VALIDAÇÃO
# =============================================================================

CONFIG = {
    "batch_size": 64,
    "epochs": 3, # Ajuste conforme necessário
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": 224,
    "use_autocast": False, # Changed to False for debugging nan loss
    "cam_fusion_mode": "learned_forward"  # Escolhas: mean, fixed_weighted, best_layer, learned_forward
}

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum()
    union = pred_bin.sum() + true_mask.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0

    pbar = tqdm(loader, desc="Treinando", leave=False)
    for images, targets_cls, targets_seg in pbar:
        images = images.to(device)
        targets_cls = targets_cls.to(device)
        targets_seg = targets_seg.to(device)

        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_autocast'], device_type=device):
            y_cls, y_seg, saliency = model(images)
            loss, _ = criterion(y_cls, targets_cls, y_seg, targets_seg, saliency)

        # Check for nan loss and skip backward pass if found
        if torch.isnan(loss):
            print("WARNING: NaN loss detected. Skipping backward pass and optimizer step.")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds_cls = torch.argmax(y_cls, dim=1)
        all_preds_cls.extend(preds_cls.cpu().numpy())
        all_targets_cls.extend(targets_cls.cpu().numpy())
        total_iou += calculate_iou(y_seg, targets_seg).item()
        pbar.set_postfix({'loss': loss.item()})

    return running_loss/len(loader), accuracy_score(all_targets_cls, all_preds_cls), total_iou/len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0

    with torch.no_grad():
        for images, targets_cls, targets_seg in tqdm(loader, desc="Validando", leave=False):
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_seg = targets_seg.to(device)

            with autocast(enabled=CONFIG['use_autocast'], device_type=device):
              y_cls, y_seg, saliency = model(images)
              loss, _ = criterion(y_cls, targets_cls, y_seg, targets_seg, saliency)
            running_loss += loss.item()

            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())
            total_iou += calculate_iou(y_seg, targets_seg).item()

    return running_loss/len(loader), accuracy_score(all_targets_cls, all_preds_cls), f1_score(all_targets_cls, all_preds_cls, average='binary'), total_iou/len(loader)

# =============================================================================
# 5. PIPELINE PRINCIPAL (LOOP)
# =============================================================================

def run_training_pipeline(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []

    print(f"Rodando em: {CONFIG['device']}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], df['group'])):
        print(f"\n{'='*20} FOLD {fold+1}/{n_splits} {'='*20}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = DataLoader(CrackDataset(train_df, img_size=CONFIG['img_size'], augment=True), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(CrackDataset(val_df, img_size=CONFIG['img_size'], augment=False), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        # Instancia o modelo AGORA que a classe está definida no topo do script
        model = MultiTaskNetwork(num_classes_cls=2, fusion_mode=CONFIG['cam_fusion_mode']).to(CONFIG['device'])
        # Otimização adicional com torch.compile se disponível (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        criterion = MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=0.5).to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Decaimento gradual

        scaler = GradScaler()

        best_f1 = 0.0

        for epoch in range(CONFIG['epochs']):
            t_loss, t_acc, t_iou = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], scaler)
            v_loss, v_acc, v_f1, v_iou = validate_one_epoch(model, val_loader, criterion, CONFIG['device'])

            print(f"Epoch {epoch+1} | T_Loss: {t_loss:.3f} T_Acc: {t_acc:.3f} T_IoU: {t_iou:.3f} | V_Acc: {v_acc:.3f} V_F1: {v_f1:.3f} V_IoU: {v_iou:.3f}")

            if v_f1 > best_f1:
                best_f1 = v_f1
                # Salva o checkpoint (opcional)
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")

            scheduler.step()  # Atualiza o scheduler após cada epoch

        fold_scores.append(best_f1)

    print(f"\nMédia F1: {np.mean(fold_scores):.4f}")

# =============================================================================
# 6. EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    # Caminho onde você salvou as imagens com o script de geração
    DATASET_DIR = 'processed_dataset_MTL'

    if os.path.exists(DATASET_DIR):
        # 1. Cria o DataFrame
        df = prepare_dataframe(DATASET_DIR)
        print(f"DataFrame carregado com {len(df)} imagens.")

        # 2. Roda o Pipeline
        if len(df) > 0:
            run_training_pipeline(df, n_splits=5)
        else:
            print("DataFrame vazio. Verifique se o dataset foi gerado corretamente.")
    else:
        print(f"Pasta {DATASET_DIR} não encontrada. Rode o script de 'create_balanced_dataset' primeiro.")

# =============================================================================
# 7. VISUALIZAÇÃO DOS RESULTADOS (CAM / SALIENCY)
# =============================================================================

def unnormalize_image(tensor):
    """Reverte a normalização do ImageNet para visualização correta."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Move para CPU e converte para numpy
    img = tensor.permute(1, 2, 0).cpu().numpy()

    # Desfaz a normalização
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_model_predictions(model, loader, device, num_images=5):
    """
    Gera um plot comparativo: Imagem | Mask GT | Mask Pred | Saliency Map
    """
    model.eval()
    images, targets_cls, targets_seg = next(iter(loader))

    images = images.to(device)

    with torch.no_grad():
        # Forward pass
        y_cls, y_seg, saliency_map = model(images)

        # Processa a segmentação (Sigmoid > 0.5)
        preds_seg = (torch.sigmoid(y_seg) > 0.5).float()

        # Upsample do Saliency Map (de 7x7 para 224x224) para visualização
        saliency_resized = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=True)

        # Normaliza o Saliency para 0-1 para plotagem limpa
        B = saliency_resized.shape[0]
        s_min = saliency_resized.view(B, -1).min(1, keepdim=True)[0].view(B, 1, 1, 1)
        s_max = saliency_resized.view(B, -1).max(1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency_resized = (saliency_resized - s_min) / (s_max - s_min + 1e-8)

    # Plotagem
    plt.figure(figsize=(16, 4 * num_images))

    for i in range(min(num_images, images.size(0))):
        # 1. Imagem Original
        plt.subplot(num_images, 4, i * 4 + 1)
        img_show = unnormalize_image(images[i])
        plt.imshow(img_show)
        cls_pred_idx = torch.argmax(y_cls[i]).item()
        cls_true_idx = targets_cls[i].item()
        label_text = "Positivo" if cls_pred_idx == 1 else "Negativo"
        color = 'green' if cls_pred_idx == cls_true_idx else 'red'
        plt.title(f"Img Original\nPred: {label_text}", color=color)
        plt.axis('off')

        # 2. Máscara Real (Ground Truth)
        plt.subplot(num_images, 4, i * 4 + 2)
        plt.imshow(targets_seg[i].cpu().squeeze(), cmap='gray')
        plt.title("Ground Truth (Mask)")
        plt.axis('off')

        # 3. Máscara Predita
        plt.subplot(num_images, 4, i * 4 + 3)
        plt.imshow(preds_seg[i].cpu().squeeze(), cmap='gray')
        plt.title("Segmentação Predita")
        plt.axis('off')

        # 4. Saliency Map (CAM)
        plt.subplot(num_images, 4, i * 4 + 4)
        # Plota a imagem original de fundo
        plt.imshow(img_show)
        # Plota o mapa de calor por cima (com transparência alpha)
        # cmap='jet' é clássico para mapas de calor (azul=frio/fundo, vermelho=quente/foco)
        plt.imshow(saliency_resized[i].cpu().squeeze(), cmap='jet', alpha=0.5)
        plt.title("Saliency Map (CAM)\n(Onde a rede olhou)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()