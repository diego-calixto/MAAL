"""Multi-task crack detection with learned attention supervision.

Combines classification, segmentation, and spatial attention regularization for robust crack evidence learning.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from typing import Tuple, Dict, List, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from torchvision import models, transforms
try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

# Enable TF32 for better performance on A100
torch.set_float32_matmul_precision('high')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Image preprocessing
IMAGE_SIZE: int = 384
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406])
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225])

# Model architecture
NUM_CLASSES_CLS: int = 2
NUM_CLASSES_SEG: int = 1
BACKBONE_CHANNELS: int = 2048

# Attention supervision stage
# 2 -> 28x28, 3 -> 14x14, 4 -> 7x7 for input 384
ATTENTION_FEATURE_LEVEL: int = 3

# Loss weights
LOSS_WEIGHT_CLS: float = 1.0
LOSS_WEIGHT_SEG: float = 1.0
LOSS_WEIGHT_ATTENTION: float = 0.5

# Segmentation loss options
# options: 'bce', 'dice', 'bce_dice', 'focal_tversky'
SEGMENTATION_LOSS_TYPE: str = "focal_tversky"
SEGMENTATION_POS_WEIGHT: float = 20.0
FOCAL_TVERSKY_ALPHA: float = 0.3
FOCAL_TVERSKY_BETA: float = 0.7
FOCAL_TVERSKY_GAMMA: float = 1.0

# Training hyperparameters
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 8
N_SPLITS: int = 5
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset constants
VALID_IMAGE_EXTENSIONS: set = {'.jpg', '.jpeg', '.png'}
DATASET_DIR: str = 'processed_dataset_MTL'
MASK_BINARY_THRESHOLD: float = 0.5

# Visualization constants
VISUALIZATION_COLORMAP: str = 'jet'
HEATMAP_COLORMAP: str = 'hot'
ATTENTION_ALPHA: float = 0.5
EPSILON: float = 1e-8

# =============================================================================
# 1. MODEL ARCHITECTURE
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
        # Simplified U-Net style decoder
        self.up4 = self.conv_block(2048 + 1024, 1024)
        self.up3 = self.conv_block(1024 + 512, 512)
        self.up2 = self.conv_block(512 + 256, 256)

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )

    def match_tensor(self, tensor_target, tensor_ref):
        if tensor_target.size()[2:] != tensor_ref.size()[2:]:
            return F.interpolate(tensor_target, size=tensor_ref.size()[2:], mode='bilinear', align_corners=True)
        return tensor_target

    def forward(self, features):
        d4 = self.up4(torch.cat([features[4], self.match_tensor(features[3], features[4])], dim=1))
        d3 = self.up3(torch.cat([d4, self.match_tensor(features[2], d4)], dim=1))
        d2 = self.up2(torch.cat([d3, self.match_tensor(features[1], d3)], dim=1))
        out_seg = self.final_conv(d2)
        return out_seg


class AttentionHead(nn.Module):
    def __init__(self, in_channels=BACKBONE_CHANNELS):
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
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss.mean()


class MultiTaskNetwork(nn.Module):
    def __init__(self, num_classes_cls=2):
        super(MultiTaskNetwork, self).__init__()
        self.encoder = SharedEncoder()

        attention_feature_channels = {
            2: 512,
            3: 1024,
            4: 2048
        }.get(ATTENTION_FEATURE_LEVEL, BACKBONE_CHANNELS)
        self.attention_head = AttentionHead(in_channels=attention_feature_channels)

        # Classification branch uses attention-gated encoder features.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(BACKBONE_CHANNELS, num_classes_cls)

        self.decoder = SegmentationDecoder(num_classes=1)

    def forward(self, x):
        features = self.encoder(x)
        f_final = features[-1]

        attention_features = features[ATTENTION_FEATURE_LEVEL]
        attention_map = self.attention_head(attention_features)
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

        # Segmentation branch remains a separate structured decoder.
        y_seg = self.decoder(features)

        return y_cls, y_seg, attention_map

# =============================================================================
# 2. LOSS FUNCTION
# =============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        w_cls=1.0,
        w_seg=1.0,
        w_attn=0.5,
        seg_loss_type="bce",
        seg_pos_weight=1.0,
        ft_alpha=0.7,
        ft_beta=0.3,
        ft_gamma=0.75
    ):
        super(MultiTaskLoss, self).__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_attn = w_attn
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_loss_type = seg_loss_type
        self.attn_criterion = nn.BCELoss()

        if self.seg_loss_type == "focal_tversky":
            self.seg_criterion = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
        else:
            pos_weight = torch.tensor([seg_pos_weight], device=DEVICE) if seg_pos_weight is not None else None
            self.seg_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, y_cls_pred, y_cls_true, y_seg_pred, y_seg_true, attention_map):
        # Classification loss
        loss_cls = self.cls_criterion(y_cls_pred, y_cls_true)

        # Segmentation losses
        y_seg_target = F.interpolate(
            y_seg_true,
            size=y_seg_pred.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        y_seg_target = y_seg_target.float()

        pred_probs = torch.sigmoid(y_seg_pred)
        intersection = (pred_probs * y_seg_target).sum(dim=(1, 2, 3))
        union = pred_probs.sum(dim=(1, 2, 3)) + y_seg_target.sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss_dice = 1.0 - dice_score
        loss_dice = loss_dice.mean()

        loss_bce = torch.tensor(0.0, device=y_seg_pred.device)

        # Apply selected segmentation loss type
        if self.seg_loss_type == "bce":
            loss_bce = self.seg_criterion(y_seg_pred, y_seg_target)
            loss_seg = loss_bce
        elif self.seg_loss_type == "dice":
            loss_seg = loss_dice
        elif self.seg_loss_type == "bce_dice":
            loss_bce = self.seg_criterion(y_seg_pred, y_seg_target)
            loss_seg = loss_bce + loss_dice
        elif self.seg_loss_type == "focal_tversky":
            loss_seg = self.seg_criterion(y_seg_pred, y_seg_target)
            loss_dice = torch.tensor(0.0, device=y_seg_pred.device)
        else:
            raise ValueError(f"Unknown seg_loss_type: {self.seg_loss_type}")

        # Attention supervision encourages the model to focus on crack regions.
        attention_target = F.interpolate(
            y_seg_target,
            size=attention_map.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        attention_loss_bce = self.attn_criterion(attention_map, attention_target)

        attn_intersection = (attention_map * attention_target).sum(dim=(1, 2, 3))
        attn_union = attention_map.sum(dim=(1, 2, 3)) + attention_target.sum(dim=(1, 2, 3))
        attn_dice = 1.0 - (2.0 * attn_intersection + EPSILON) / (attn_union + EPSILON)
        attn_dice = attn_dice.mean()

        loss_attention = attention_loss_bce + attn_dice
        total_loss = (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_attn * loss_attention)

        return total_loss, {
            "cls": loss_cls,
            "seg_bce": loss_bce,
            "seg_dice": loss_dice,
            "attention_bce": attention_loss_bce,
            "attention_dice": attn_dice
        }

# =============================================================================
# 3. DATASET AND DATA LOADING
# =============================================================================

class CrackDataset(Dataset):
    def __init__(self, df, img_size=224, augment=False):
        self.df = df
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = cv2.imread(row['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > MASK_BINARY_THRESHOLD).astype(np.float32)

        # Resize both image and mask
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
    """Load dataset directory structure into pandas DataFrame."""
    data = []
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
            if ext not in VALID_IMAGE_EXTENSIONS:
                continue

            group_id = entry.name.split('_')[0]
            mask_name = os.path.splitext(entry.name)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"WARNING: Mask not found for {entry.path}. Using zero mask.")

            data.append({
                'path': entry.path,
                'mask_path': mask_path,
                'filename': entry.name,
                'label': label,
                'group': group_id
            })
    return pd.DataFrame(data)

# =============================================================================
# 4. TRAINING AND VALIDATION FUNCTIONS
# =============================================================================

CONFIG = {
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "lr": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "device": DEVICE,
    "img_size": IMAGE_SIZE,
    "use_autocast": False,
    "use_compile": False,
    "attention_weight": LOSS_WEIGHT_ATTENTION
}


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate Intersection over Union metric per image in batch, then average."""
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou.mean()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets_cls, targets_seg in pbar:
        images = images.to(device)
        targets_cls = targets_cls.to(device)
        targets_seg = targets_seg.to(device)

        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_autocast'], device_type=device):
            y_cls, y_seg, attention_map = model(images)
            loss, _ = criterion(y_cls, targets_cls, y_seg, targets_seg, attention_map)

        if torch.isnan(loss):
            print("WARNING: NaN loss detected. Skipping batch.")
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
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0

    with torch.no_grad():
        for images, targets_cls, targets_seg in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_seg = targets_seg.to(device)

            with autocast(enabled=CONFIG['use_autocast'], device_type=device):
                y_cls, y_seg, attention_map = model(images)
                loss, _ = criterion(y_cls, targets_cls, y_seg, targets_seg, attention_map)
            running_loss += loss.item()

            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())
            total_iou += calculate_iou(y_seg, targets_seg).item()

    return (running_loss/len(loader), accuracy_score(all_targets_cls, all_preds_cls), 
            f1_score(all_targets_cls, all_preds_cls, average='binary'), total_iou/len(loader))


def classification_per_class_iou(model, loader, device, num_classes=2):
    """Compute per-class IoU for classification predictions."""
    model.eval()
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, targets_cls, _ in loader:
            images = images.to(device)
            targets = targets_cls.to(device)
            logits, _, _ = model(images)
            preds = torch.argmax(logits, dim=1)

            for c in range(num_classes):
                pred_c = (preds == c)
                true_c = (targets == c)
                tp[c] += int((pred_c & true_c).sum().cpu().item())
                fp[c] += int((pred_c & ~true_c).sum().cpu().item())
                fn[c] += int((~pred_c & true_c).sum().cpu().item())

    ious = {}
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        ious[c] = (tp[c] / denom) if denom > 0 else 0.0
    return ious


def visualize_predictions(model, loader, device, out_dir='visuals', n_samples=16, threshold=0.5):
    """Save a set of visualization images: input, GT mask, pred mask, attention overlay."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    inv_mean = -IMAGENET_MEAN / IMAGENET_STD
    inv_std = 1.0 / IMAGENET_STD

    with torch.no_grad():
        for images, targets_cls, targets_seg in loader:
            images = images.to(device)
            targets_seg = targets_seg.to(device)

            logits, y_seg, attention_map = model(images)
            probs = torch.sigmoid(y_seg)
            preds_mask = (probs > threshold).float()

            for i in range(images.shape[0]):
                if saved >= n_samples:
                    return

                img_t = images[i].cpu()
                # unnormalize
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

                # attention overlay
                hm = plt.cm.get_cmap(HEATMAP_COLORMAP)(attn_np)[:, :, :3]
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

    return

# =============================================================================
# 5. TRAINING PIPELINE
# =============================================================================

def run_training_pipeline(df, n_splits=5):
    """Run multi-fold cross-validation training."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []

    print(f"Running on: {CONFIG['device']}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], df['group'])):
        print(f"\n{'='*20} FOLD {fold+1}/{n_splits} {'='*20}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = DataLoader(
            CrackDataset(train_df, img_size=CONFIG['img_size'], augment=True), 
            batch_size=CONFIG['batch_size'], shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            CrackDataset(val_df, img_size=CONFIG['img_size'], augment=False), 
            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        model = MultiTaskNetwork(num_classes_cls=2).to(CONFIG['device'])
        if CONFIG.get('use_compile', False) and hasattr(torch, 'compile'):
            model = torch.compile(model)
        criterion = MultiTaskLoss(
            w_cls=1.0,
            w_seg=1.0,
            w_attn=CONFIG['attention_weight'],
            seg_loss_type=SEGMENTATION_LOSS_TYPE,
            seg_pos_weight=SEGMENTATION_POS_WEIGHT,
            ft_alpha=FOCAL_TVERSKY_ALPHA,
            ft_beta=FOCAL_TVERSKY_BETA,
            ft_gamma=FOCAL_TVERSKY_GAMMA
        ).to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scaler = GradScaler()

        # LR scheduler (reduce LR when val IoU plateaus) and early stopping
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_f1 = 0.0
        best_v_iou = 0.0
        es_patience = 6
        es_wait = 0

        for epoch in range(CONFIG['epochs']):
            t_loss, t_acc, t_iou = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], scaler)
            v_loss, v_acc, v_f1, v_iou = validate_one_epoch(model, val_loader, criterion, CONFIG['device'])

            print(f"Epoch {epoch+1} | T_Loss: {t_loss:.3f} T_Acc: {t_acc:.3f} T_IoU: {t_iou:.3f} | "
                  f"V_Acc: {v_acc:.3f} V_F1: {v_f1:.3f} V_IoU: {v_iou:.3f}")

            # Step scheduler with validation IoU (we want to maximize IoU)
            try:
                scheduler.step(v_iou)
            except Exception:
                # Fallback: if scheduler expects loss, pass -v_iou
                scheduler.step(-v_iou)

            # Early stopping based on validation IoU
            if v_iou > best_v_iou:
                best_v_iou = v_iou
                es_wait = 0
                # keep best-f1 as well for legacy reporting
                if v_f1 > best_f1:
                    best_f1 = v_f1
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
            else:
                es_wait += 1
                if es_wait >= es_patience:
                    print(f"Early stopping triggered (no improvement in {es_patience} epochs).")
                    break

        fold_scores.append(best_v_iou)

        # After fold training, load best model and compute diagnostics + visuals
        best_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(best_path):
            print(f"Loading best model for fold {fold} from {best_path} for diagnostics...")
            model.load_state_dict(torch.load(best_path, map_location=CONFIG['device']))
            # Per-class classification IoU
            per_class_iou = classification_per_class_iou(model, val_loader, CONFIG['device'], num_classes=2)
            print(f"Fold {fold} per-class classification IoU: {per_class_iou}")

            # Save visualizations
            vis_out = os.path.join('visuals', f'fold_{fold}')
            print(f"Saving sample visualizations to {vis_out}")
            visualize_predictions(model, val_loader, CONFIG['device'], out_dir=vis_out, n_samples=32)
        else:
            print(f"Best model file {best_path} not found; skipping diagnostics for fold {fold}.")

    print(f"\nMean Val IoU: {np.mean(fold_scores):.4f}")

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        df = prepare_dataframe(DATASET_DIR)
        print(f"DataFrame loaded with {len(df)} images.")

        if len(df) > 0:
            run_training_pipeline(df, n_splits=N_SPLITS)
        else:
            print("DataFrame empty. Check if dataset was generated correctly.")
    else:
        print(f"Dataset directory {DATASET_DIR} not found. Run dataset creation script first.")
