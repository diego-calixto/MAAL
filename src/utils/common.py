import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# Shared constants for both model experiments
# Optimized for 80GB A100 GPU
IMAGE_SIZE = 384  # Can increase to 448 or 512 if desired
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
BATCH_SIZE = 32  # Increased from 32 (80GB VRAM allows this)
NUM_EPOCHS = 30  # Increased from 10 for better convergence
LEARNING_RATE = 7.5e-5  # Slightly increased due to larger batch size
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "processed_dataset_MTL"
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MASK_BINARY_THRESHOLD = 0.5
USE_AUTOCAST = True  # Disabled - causes issues on some systems
USE_COMPILE = False  # Disabled - optional optimization
ALIGNMENT_WEIGHT = 0.05
EPSILON = 1e-8


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
        self.up4 = self.conv_block(2048 + 1024, 1024)
        self.up3 = self.conv_block(1024 + 512, 512)
        self.up2 = self.conv_block(512 + 256, 256)

        self.final_conv = nn.Sequential(
            # 1. Faz a convolução 3x3 pesada na resolução menor (96x96)
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # 2. Faz o upsampling do tensor já leve (apenas 64 canais)
            nn.Upsample(
                scale_factor=4,
                mode='bilinear',
                align_corners=True
            ), # Shape resultante aqui: [64, 64, 384, 384]
            
            # 3. Projeta para 1 canal final
            nn.Conv2d(64, 1, 1)
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


class CrackDataset(Dataset):
    def __init__(self, df, img_size=IMAGE_SIZE, augment=False):
        self.df = df
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

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

    for label_name in ['Positive', 'Negative']:
        # Caminho aponta para a pasta de IMAGENS
        class_dir_img = os.path.join(dataset_dir, label_name, 'images')

        if not os.path.exists(class_dir_img):
            continue

        label = 1 if label_name == 'Positive' else 0

        for entry in os.scandir(class_dir_img):
            if entry.name.endswith('.jpg'):
                # group_id para o Cross Validation (evita vazamento de dados)
                group_id = entry.name.split('_')[0]

                # O caminho da máscara é inferido trocando pastas e extensão
                # Ex: .../Positive/images/foto.jpg -> .../Positive/masks/foto.png
                mask_path = entry.path.replace('images', 'masks').replace('.jpg', '.png')

                data.append({
                    'path': entry.path,
                    'mask_path': mask_path,
                    'filename': entry.name,
                    'label': label,
                    'group': group_id
                })

    return pd.DataFrame(data)


def save_checkpoint(checkpoint: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None, device=DEVICE):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scaler is not None and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])

    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    return checkpoint


if __name__ == '__main__':
    df = prepare_dataframe('processed_dataset_MTL')
    print(f"Registros criados: {len(df)}")
    if len(df) > 0:
        print("Exemplo de caminho imagem:", df.iloc[0]['path'])
        print("Exemplo de caminho máscara:", df.iloc[0]['mask_path'])


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou.mean()


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold = 0.5, from_logits: bool = False) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred_mask) if from_logits else pred_mask
    if threshold is not None:
        pred_bin = (pred_prob > threshold).float()
    else:
        pred_bin = pred_prob
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    volume = pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + EPSILON) / (volume + EPSILON)
    return dice.mean()


def calculate_binary_iou(pred_bin: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou.mean()


def calculate_pixel_accuracy(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    correct = (pred_bin == true_mask).float()
    return correct.mean()


def calculate_pointing_game(saliency_map: torch.Tensor, true_mask: torch.Tensor) -> float:
    """
    Calculates Pointing Game hit rate.
    Returns the ratio of hits (where the max pixel in saliency_map falls inside true_mask == 1).
    """
    B = saliency_map.size(0)
    hits = 0.0
    for i in range(B):
        sal = saliency_map[i, 0]
        mask = true_mask[i, 0]
        
        # Avoid empty mask issues if checking
        if mask.max() == 0:
            continue # no object to point to
            
        max_idx = torch.argmax(sal).item()
        H, W = sal.shape
        y = max_idx // W
        x = max_idx % W
        
        if mask[y, x] > 0.5:
            hits += 1.0
            
    # Compute over instances that actually have an object
    num_objects = sum([1 for i in range(B) if true_mask[i, 0].max() > 0])
    return hits / max(1, num_objects)


def calculate_insertion_deletion_score(model, image: torch.Tensor, saliency_map: torch.Tensor, target_cls: int, steps: int = 10, mode: str = 'deletion'):
    """
    Standalone function to compute Deletion or Insertion score for a single image.
    image: [1, 3, H, W]
    saliency_map: [1, 1, H, W] or [H, W]
    mode: 'deletion' or 'insertion'
    """
    model.eval()
    if saliency_map.dim() == 4:
        saliency_map = saliency_map[0, 0]
    elif saliency_map.dim() == 3:
        saliency_map = saliency_map[0]
        
    with torch.no_grad():
        out_cls = model(image)
        if isinstance(out_cls, (list, tuple)):
            out_cls = out_cls[0]
        probs = torch.softmax(out_cls, dim=1)
        initial_conf = probs[0, target_cls].item()

    flat_saliency = saliency_map.flatten()
    sorted_indices = torch.argsort(flat_saliency, descending=True)
    num_pixels = flat_saliency.size(0)
    pixels_per_step = num_pixels // steps
    
    scores = [initial_conf]
    
    if mode == 'deletion':
        perturbed_img = image.clone()
        fill_value = 0.0 # Or image mean
    elif mode == 'insertion':
        perturbed_img = torch.zeros_like(image)
        
    for step in range(1, steps + 1):
        idx_start = (step - 1) * pixels_per_step
        idx_end = step * pixels_per_step if step < steps else num_pixels
        
        current_indices = sorted_indices[idx_start:idx_end]
        H, W = saliency_map.shape
        y = current_indices // W
        x = current_indices % W
        
        if mode == 'deletion':
            perturbed_img[0, :, y, x] = fill_value
        elif mode == 'insertion':
            perturbed_img[0, :, y, x] = image[0, :, y, x]
            
        with torch.no_grad():
            out_cls = model(perturbed_img)
            if isinstance(out_cls, (list, tuple)):
                out_cls = out_cls[0]
            probs = torch.softmax(out_cls, dim=1)
            scores.append(probs[0, target_cls].item())
            
    auc = np.trapz(scores, dx=1.0/steps)
    return auc, scores


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch=0):
    model.train()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    else:
        criterion._epoch = epoch  # For sanity check printing in first epoch
        criterion.batch_count = 0  # Reset batch count for sanity check
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    loss_components = {"cls": 0.0, "seg": 0.0, "align_bce": 0.0, "align_dice": 0.0, "align": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets_cls, targets_seg in pbar:
        images = images.to(device)
        targets_cls = targets_cls.to(device)
        targets_seg = targets_seg.to(device)

        optimizer.zero_grad()
        with torch.autocast(enabled=USE_AUTOCAST, device_type='cuda' if device == 'cuda' else 'cpu'):
            outputs = model(images)
            # Support models that return 2 (cls, seg), 3 (cls, seg, cam) or 4 outputs (cls, seg, cam, saliency_maps)
            if isinstance(outputs, (list, tuple)):
                if len(outputs) == 4:
                    y_cls, y_seg, cam, _saliency_maps = outputs
                elif len(outputs) == 3:
                    y_cls, y_seg, cam = outputs
                    _saliency_maps = []
                else:
                    y_cls, y_seg = outputs
                    cam = torch.zeros_like(y_seg)
                    _saliency_maps = []
            else:
                # Unexpected single-tensor output -> treat as segmentation logits
                y_cls = torch.zeros((images.size(0), 2), device=images.device)
                y_seg = outputs
                cam = torch.zeros_like(y_seg)
                _saliency_maps = []

            loss, loss_dict = criterion(y_cls, targets_cls, y_seg, targets_seg, cam)

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss detected. Skipping batch.")
            print(f"  loss={loss}, shapes: images={tuple(images.shape)}, y_cls={tuple(y_cls.shape)}, y_seg={tuple(y_seg.shape)}, cam={tuple(cam.shape)}")
            continue

        if USE_AUTOCAST:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()
        num_batches += 1

        preds_cls = torch.argmax(y_cls, dim=1)
        all_preds_cls.extend(preds_cls.cpu().numpy())
        all_targets_cls.extend(targets_cls.cpu().numpy())
        total_iou += calculate_iou(y_seg, targets_seg).item()
        pbar.set_postfix({'loss': loss.item()})

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches

    # Log loss components (all epochs)
    print(f"Train Loss Components: cls={loss_components['cls']:.4f}, seg={loss_components['seg']:.4f}, "
          f"align_bce={loss_components['align_bce']:.4f}, align_dice={loss_components['align_dice']:.4f}, "
          f"align={loss_components['align']:.4f}")

    return running_loss / len(loader), accuracy_score(all_targets_cls, all_preds_cls), total_iou / len(loader)


def validate_one_epoch(model, loader, criterion, device, epoch=0):
    model.eval()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    else:
        criterion._epoch = epoch  # For sanity check printing if needed
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    total_dice = 0.0
    total_cam_iou = 0.0
    total_cam_dice = 0.0
    total_stage_cam_iou = []
    total_stage_cam_dice = []
    loss_components = {"cls": 0.0, "seg": 0.0, "align_bce": 0.0, "align_dice": 0.0, "align": 0.0}
    num_batches = 0

    with torch.no_grad():
        for images, targets_cls, targets_seg in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_seg = targets_seg.to(device)

            with torch.autocast(enabled=USE_AUTOCAST, device_type=device):
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    if len(outputs) == 4:
                        y_cls, y_seg, cam, saliency_maps = outputs
                    elif len(outputs) == 3:
                        y_cls, y_seg, cam = outputs
                        saliency_maps = []
                    else:
                        y_cls, y_seg = outputs
                        cam = torch.zeros_like(y_seg)
                        saliency_maps = []
                else:
                    # Unexpected single-tensor output -> treat as segmentation logits
                    y_cls = torch.zeros((images.size(0), 2), device=images.device)
                    y_seg = outputs
                    cam = torch.zeros_like(y_seg)
                    saliency_maps = []

                loss, loss_dict = criterion(y_cls, targets_cls, y_seg, targets_seg, cam)

            running_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key].item()
            num_batches += 1

            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())
            total_iou += calculate_iou(y_seg, targets_seg).item()
            total_dice += calculate_dice(y_seg, targets_seg, threshold=0.5, from_logits=True).item()

            cam_up = F.interpolate(cam, size=targets_seg.shape[2:], mode='bilinear', align_corners=True)
            cam_prob = torch.sigmoid(cam_up)
            cam_pred = (cam_prob > 0.5).float()
            total_cam_iou += calculate_binary_iou(cam_pred, targets_seg).item()
            total_cam_dice += calculate_dice(cam_prob, targets_seg, threshold=None, from_logits=False).item()

            if saliency_maps:
                if not total_stage_cam_iou:
                    total_stage_cam_iou = [0.0] * len(saliency_maps)
                    total_stage_cam_dice = [0.0] * len(saliency_maps)
                for idx, stage_map in enumerate(saliency_maps):
                    stage_up = F.interpolate(stage_map, size=targets_seg.shape[2:], mode='bilinear', align_corners=True)
                    stage_prob = torch.sigmoid(stage_up)
                    stage_pred = (stage_prob > 0.5).float()
                    total_stage_cam_iou[idx] += calculate_binary_iou(stage_pred, targets_seg).item()
                    total_stage_cam_dice[idx] += calculate_dice(stage_prob, targets_seg, threshold=None, from_logits=False).item()

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches

    # Average stage metrics and pad to 4 scales if necessary
    stage_cam_iou = [0.0] * 4
    stage_cam_dice = [0.0] * 4
    if total_stage_cam_iou:
        for idx in range(min(len(total_stage_cam_iou), 4)):
            stage_cam_iou[idx] = total_stage_cam_iou[idx] / len(loader)
            stage_cam_dice[idx] = total_stage_cam_dice[idx] / len(loader)

    # Log loss components (all epochs)
    print(f"Val Loss Components: cls={loss_components['cls']:.4f}, seg={loss_components['seg']:.4f}, "
          f"align_bce={loss_components['align_bce']:.4f}, align_dice={loss_components['align_dice']:.4f}, "
          f"align={loss_components['align']:.4f}")

    return (
        running_loss / len(loader),
        accuracy_score(all_targets_cls, all_preds_cls),
        f1_score(all_targets_cls, all_preds_cls, average='binary'),
        total_iou / len(loader),
        total_dice / len(loader),
        total_cam_iou / len(loader),
        total_cam_dice / len(loader),
        stage_cam_iou[0],
        stage_cam_iou[1],
        stage_cam_iou[2],
        stage_cam_iou[3],
        stage_cam_dice[0],
        stage_cam_dice[1],
        stage_cam_dice[2],
        stage_cam_dice[3]
    )


def run_training_pipeline(
    run_name,
    model_factory,
    criterion_factory,
    df,
    n_splits=N_SPLITS,
    checkpoint_dir='checkpoints',
    resume_from=None,
    save_every=1
):
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    pin_memory = DEVICE == 'cuda'
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"Running on: {DEVICE}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], df['group'])):
        print(f"\n{'='*20} FOLD {fold+1}/{n_splits} {'='*20}")

        fold_dir = os.path.join(run_checkpoint_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(fold_dir, 'best.pt')
        last_checkpoint_path = os.path.join(fold_dir, 'last.pt')

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = DataLoader(
            CrackDataset(train_df, img_size=IMAGE_SIZE, augment=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0
        )
        val_loader = DataLoader(
            CrackDataset(val_df, img_size=IMAGE_SIZE, augment=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0
        )

        model = model_factory().to(DEVICE)
        if USE_COMPILE and hasattr(torch, 'compile'):
            model = torch.compile(model)

        criterion = criterion_factory().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AUTOCAST)

        best_f1 = 0.0
        best_v_iou = 0.0
        es_patience = 6
        es_wait = 0
        start_epoch = 0

        if resume_from is not None and os.path.exists(resume_from):
            checkpoint = load_checkpoint(resume_from, model, optimizer, scaler, scheduler, device=DEVICE)
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_v_iou = checkpoint.get('best_v_iou', best_v_iou)
            best_f1 = checkpoint.get('best_f1', best_f1)
            es_wait = checkpoint.get('es_wait', es_wait)
            print(f"Resuming fold {fold} from checkpoint '{resume_from}' at epoch {start_epoch}.")

        for epoch in range(start_epoch, NUM_EPOCHS):
            t_loss, t_acc, t_iou = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, epoch=epoch)
            (
                v_loss,
                v_acc,
                v_f1,
                v_iou,
                v_dice,
                v_cam_iou,
                v_cam_dice,
                stage1_cam_iou,
                stage2_cam_iou,
                stage3_cam_iou,
                stage4_cam_iou,
                stage1_cam_dice,
                stage2_cam_dice,
                stage3_cam_dice,
                stage4_cam_dice,
            ) = validate_one_epoch(model, val_loader, criterion, DEVICE, epoch=epoch)

            print(
                f"Epoch {epoch+1} | T_Loss: {t_loss:.3f} T_Acc: {t_acc:.3f} T_IoU: {t_iou:.3f} | "
                f"V_Acc: {v_acc:.3f} V_F1: {v_f1:.3f} V_IoU: {v_iou:.3f} V_Dice: {v_dice:.3f} "
                f"V_CAM_IoU: {v_cam_iou:.3f} V_CAM_Dice: {v_cam_dice:.3f}"
            )
            print(
                f"    Stage CAM IoUs: S1={stage1_cam_iou:.3f}, S2={stage2_cam_iou:.3f}, "
                f"S3={stage3_cam_iou:.3f}, S4={stage4_cam_iou:.3f}"
            )
            print(
                f"    Stage CAM Soft Dice: S1={stage1_cam_dice:.3f}, S2={stage2_cam_dice:.3f}, "
                f"S3={stage3_cam_dice:.3f}, S4={stage4_cam_dice:.3f}"
            )

            if hasattr(model, 'fusion_conv'):
                weights = model.fusion_conv.weight.data.squeeze().cpu().numpy()
                print(
                    f"    Fusion weights: L1={weights[0]:.3f}, L2={weights[1]:.3f}, "
                    f"L3={weights[2]:.3f}, L4={weights[3]:.3f}"
                )

            scheduler.step(v_iou)

            is_best = v_iou > best_v_iou
            if is_best:
                best_v_iou = v_iou
                best_f1 = max(best_f1, v_f1)
                es_wait = 0
            else:
                es_wait += 1

            checkpoint_state = {
                'run_name': run_name,
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_v_iou': best_v_iou,
                'best_f1': best_f1,
                'es_wait': es_wait,
            }
            if hasattr(model, 'fusion_conv'):
                checkpoint_state['fusion_weights'] = model.fusion_conv.weight.data.cpu()

            save_checkpoint(checkpoint_state, last_checkpoint_path)
            if is_best:
                save_checkpoint(checkpoint_state, best_checkpoint_path)
                print(f"Saved best checkpoint: {best_checkpoint_path}")

            if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch == NUM_EPOCHS - 1):
                epoch_checkpoint_path = os.path.join(fold_dir, f'epoch_{epoch+1:03d}.pt')
                save_checkpoint(checkpoint_state, epoch_checkpoint_path)

            if es_wait >= es_patience:
                print(f"Early stopping after {es_patience} epochs without improvement.")
                break

        fold_scores.append(best_v_iou)

    print(f"\nMean Val IoU: {np.mean(fold_scores):.4f}")
    return fold_scores
