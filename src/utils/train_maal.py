#!/usr/bin/env python3
"""
MAAL Multi-Task Training Script
Specialized training script for MAAL (Multi-scale Adaptive Alignment Loss) model.
Handles per-scale saliency alignment with learnable adaptive weights.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.common import (
    IMAGE_SIZE,
    DEVICE,
    DATASET_DIR,
    N_SPLITS,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_AUTOCAST,
    USE_COMPILE,
    ALIGNMENT_WEIGHT,
    CrackDataset,
    prepare_dataframe,
    calculate_iou,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    'train_one_epoch_maal',
    'validate_one_epoch_maal',
    'run_maal_training_pipeline',
]


def train_one_epoch_maal(model, loader, optimizer, criterion, device, scaler, epoch=0):
    """
    Train one epoch with MAAL loss.
    Passes individual saliency_maps (not fused) to the loss function.
    """
    model.train()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    loss_components = {"cls": 0.0, "seg": 0.0, "align_bce": 0.0, "align_dice": 0.0, "align_maal": 0.0}
    alpha_weights_sum = [0.0] * 4  # Track average alphas per scale
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets_cls, targets_seg in pbar:
        images = images.to(device)
        targets_cls = targets_cls.to(device)
        targets_seg = targets_seg.to(device)

        optimizer.zero_grad()
        with torch.autocast(enabled=USE_AUTOCAST, device_type='cuda' if device == 'cuda' else 'cpu'):
            # MAAL network returns: (y_cls, y_seg, fused_map, saliency_maps_list)
            y_cls, y_seg, fused, saliency_maps = model(images)
            
            # Pass individual saliency_maps to MAAL loss (not the fused map)
            loss, loss_dict = criterion(y_cls, targets_cls, y_seg, targets_seg, saliency_maps)

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss detected. Skipping batch.")
            print(f"  loss={loss}, shapes: images={tuple(images.shape)}, y_cls={tuple(y_cls.shape)}, y_seg={tuple(y_seg.shape)}")
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
            if key in loss_dict:
                loss_components[key] += loss_dict[key].item()
        
        # Track alphas for monitoring
        if 'alphas' in loss_dict:
            alphas_np = loss_dict['alphas'].cpu().numpy()
            for i in range(min(4, len(alphas_np))):
                alpha_weights_sum[i] += alphas_np[i]
        
        num_batches += 1

        preds_cls = torch.argmax(y_cls, dim=1)
        all_preds_cls.extend(preds_cls.cpu().numpy())
        all_targets_cls.extend(targets_cls.cpu().numpy())
        total_iou += calculate_iou(y_seg, targets_seg).item()
        pbar.set_postfix({'loss': loss.item()})

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches
    
    # Average alpha weights
    avg_alphas = [w / num_batches for w in alpha_weights_sum]

    # Log loss components and alphas
    print(f"Train Loss Components: cls={loss_components['cls']:.4f}, seg={loss_components['seg']:.4f}, "
          f"align_bce={loss_components['align_bce']:.4f}, align_dice={loss_components['align_dice']:.4f}, "
          f"align_maal={loss_components['align_maal']:.4f}")
    print(f"Train Avg Alphas (per-scale): S1={avg_alphas[0]:.4f}, S2={avg_alphas[1]:.4f}, "
          f"S3={avg_alphas[2]:.4f}, S4={avg_alphas[3]:.4f}")

    return running_loss / len(loader), accuracy_score(all_targets_cls, all_preds_cls), total_iou / len(loader), avg_alphas


def validate_one_epoch_maal(model, loader, criterion, device, epoch=0):
    """
    Validate one epoch with MAAL loss.
    Passes individual saliency_maps to the loss function.
    """
    model.eval()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    
    val_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    loss_components = {"cls": 0.0, "seg": 0.0, "align_bce": 0.0, "align_dice": 0.0, "align_maal": 0.0}
    alpha_weights_sum = [0.0] * 4
    num_batches = 0

    with torch.no_grad():
        for images, targets_cls, targets_seg in loader:
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_seg = targets_seg.to(device)

            with torch.autocast(enabled=USE_AUTOCAST, device_type='cuda' if device == 'cuda' else 'cpu'):
                y_cls, y_seg, fused, saliency_maps = model(images)
                loss, loss_dict = criterion(y_cls, targets_cls, y_seg, targets_seg, saliency_maps)

            if torch.isfinite(loss):
                val_loss += loss.item()
                for key in loss_components:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key].item()
                
                if 'alphas' in loss_dict:
                    alphas_np = loss_dict['alphas'].cpu().numpy()
                    for i in range(min(4, len(alphas_np))):
                        alpha_weights_sum[i] += alphas_np[i]
                
                num_batches += 1

            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())
            total_iou += calculate_iou(y_seg, targets_seg).item()

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches
    
    # Average alpha weights
    avg_alphas = [w / num_batches for w in alpha_weights_sum]

    val_loss /= len(loader)
    val_acc = accuracy_score(all_targets_cls, all_preds_cls)
    val_iou = total_iou / len(loader)

    return val_loss, val_acc, val_iou, loss_components, avg_alphas


def run_maal_training_pipeline(
    model_factory,
    criterion_factory,
    df,
    n_splits=N_SPLITS,
    checkpoint_dir='checkpoints',
    resume_from=None,
):
    """
    Run MAAL training with k-fold cross-validation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    pin_memory = DEVICE == 'cuda'
    run_checkpoint_dir = os.path.join(checkpoint_dir, 'maal')
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"Running MAAL training on: {DEVICE}")

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
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AUTOCAST)

        best_iou = 0.0
        es_patience = 6
        es_wait = 0
        start_epoch = 0

        if resume_from is not None and os.path.exists(resume_from):
            checkpoint = load_checkpoint(resume_from, model, optimizer, scaler, scheduler, device=DEVICE)
            # Load criterion state (MAAL alpha_weights)
            if 'criterion_state' in checkpoint:
                criterion.load_state_dict(checkpoint['criterion_state'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_iou = checkpoint.get('best_iou', best_iou)
            es_wait = checkpoint.get('es_wait', es_wait)
            print(f"Resuming fold {fold} from checkpoint '{resume_from}' at epoch {start_epoch}.")

        for epoch in range(start_epoch, NUM_EPOCHS):
            t_loss, t_acc, t_iou, t_alphas = train_one_epoch_maal(
                model, train_loader, optimizer, criterion, DEVICE, scaler, epoch=epoch
            )
            
            v_loss, v_acc, v_iou, v_loss_dict, v_alphas = validate_one_epoch_maal(
                model, val_loader, criterion, DEVICE, epoch=epoch
            )

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | T_Loss: {t_loss:.3f} T_Acc: {t_acc:.3f} T_IoU: {t_iou:.3f} | "
                f"V_Loss: {v_loss:.3f} V_Acc: {v_acc:.3f} V_IoU: {v_iou:.3f}"
            )
            print(
                f"    Val Alphas: S1={v_alphas[0]:.4f}, S2={v_alphas[1]:.4f}, "
                f"S3={v_alphas[2]:.4f}, S4={v_alphas[3]:.4f}"
            )

            scheduler.step(v_iou)

            is_best = v_iou > best_iou
            if is_best:
                best_iou = v_iou
                es_wait = 0
            else:
                es_wait += 1

            checkpoint_state = {
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'criterion_state': criterion.state_dict(),  # Save MAAL alpha_weights
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_iou': best_iou,
                'es_wait': es_wait,
            }

            save_checkpoint(checkpoint_state, last_checkpoint_path)
            if is_best:
                save_checkpoint(checkpoint_state, best_checkpoint_path)
                print(f"Saved best checkpoint: {best_checkpoint_path}")

            if es_wait >= es_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        print(f"Fold {fold+1} completed. Best Val IoU: {best_iou:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train MAAL multi-task model with checkpoint support.')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoint files')
    parser.add_argument('--resume-from', default=None, help='Path to a checkpoint file to resume training from')
    args = parser.parse_args()

    if os.path.exists(DATASET_DIR):
        df = prepare_dataframe(DATASET_DIR)
        print(f"DataFrame loaded with {len(df)} images.")

        if len(df) > 0:
            from src.models.maal import MultiTaskNetwork, MultiTaskLoss

            run_maal_training_pipeline(
                model_factory=lambda: MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward'),
                criterion_factory=lambda: MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT, num_scales=4),
                df=df,
                n_splits=N_SPLITS,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from
            )

            # After training, log learned MAAL weights for each fold
            for fold in range(N_SPLITS):
                best_path = os.path.join(args.checkpoint_dir, 'maal', f'fold_{fold}', 'best.pt')
                if os.path.exists(best_path):
                    ckpt = torch.load(best_path, map_location=DEVICE)
                    from src.models.maal import MultiTaskNetwork, MultiTaskLoss

                    model = MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward').to(DEVICE)
                    criterion = MultiTaskLoss(w_cls=1.0, w_seg=1.0, w_align=ALIGNMENT_WEIGHT, num_scales=4).to(DEVICE)
                    model.load_state_dict(ckpt['model_state'])
                    criterion.load_state_dict(ckpt.get('criterion_state', {}))
                    
                    try:
                        alphas = torch.nn.functional.softmax(criterion.alpha_weights, dim=0).cpu().numpy()
                        fusion_weights = model.fusion_conv.weight.data.squeeze().cpu().numpy()
                        print(f"Fold {fold} MAAL alphas: S1={alphas[0]:.4f}, S2={alphas[1]:.4f}, "
                              f"S3={alphas[2]:.4f}, S4={alphas[3]:.4f}")
                        print(f"Fold {fold} fusion weights: L1={fusion_weights[0]:.4f}, L2={fusion_weights[1]:.4f}, "
                              f"L3={fusion_weights[2]:.4f}, L4={fusion_weights[3]:.4f}")
                    except Exception as e:
                        print(f"Fold {fold}: unable to read weights - {e}")
                else:
                    print(f"No best checkpoint for fold {fold} at {best_path}")
        else:
            print("DataFrame empty. Verify that the dataset was generated correctly.")
    else:
        print(f"Folder {DATASET_DIR} not found. Run the 'create_balanced_dataset' script first.")


if __name__ == '__main__':
    main()
