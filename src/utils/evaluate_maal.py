#!/usr/bin/env python3
"""
MAAL Evaluation Script
Calculates standard metrics and XAI metrics (Insertion, Deletion) from a saved checkpoint.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.common import (
    IMAGE_SIZE,
    DEVICE,
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    CrackDataset,
    prepare_dataframe,
    calculate_iou,
    calculate_dice,
    calculate_pixel_accuracy,
    calculate_pointing_game,
    calculate_binary_iou,
    calculate_insertion_deletion_score,
    load_checkpoint,
)

def evaluate(checkpoint_path, model_module_name, num_xai_samples=50):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    # Dynamically import the model from the specified experiment module
    import importlib
    try:
        model_module = importlib.import_module(model_module_name)
        MultiTaskNetwork = model_module.MultiTaskNetwork
    except ImportError as e:
        print(f"Error importing MultiTaskNetwork from {model_module_name}: {e}")
        return

    # Prepare data
    print(f"Loading dataset from {DATASET_DIR}...")
    df = prepare_dataframe(DATASET_DIR)
    if len(df) == 0:
        print("Error: Empty dataset.")
        return

    val_loader = DataLoader(
        CrackDataset(df, img_size=IMAGE_SIZE, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == 'cuda'
    )

    # Initialize model
    print(f"Loading model from {checkpoint_path}...")
    model = MultiTaskNetwork(num_classes_cls=2, fusion_mode='learned_forward').to(DEVICE)
    
    try:
        load_checkpoint(checkpoint_path, model, device=DEVICE)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()

    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    total_dice = 0.0
    total_pix_acc = 0.0
    total_point_game = 0.0
    total_saliency_iou = 0.0
    num_batches = 0

    print("\n--- Running Standard Metrics (Classification & Segmentation) ---")
    with torch.no_grad():
        for images, targets_cls, targets_seg in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            targets_cls = targets_cls.to(DEVICE)
            targets_seg = targets_seg.to(DEVICE)

            y_cls, y_seg, fused, saliency_maps = model(images)

            # Classification
            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())

            # Segmentation
            total_iou += calculate_iou(y_seg, targets_seg).item()
            total_dice += calculate_dice(y_seg, targets_seg, threshold=0.5, from_logits=True).item()
            total_pix_acc += calculate_pixel_accuracy(y_seg, targets_seg).item()

            # Interpretation (Fast)
            fused_up = torch.nn.functional.interpolate(fused, size=targets_seg.shape[2:], mode='bilinear', align_corners=True)
            fused_prob = torch.sigmoid(fused_up)
            fused_pred = (fused_prob > 0.5).float()
            
            total_point_game += calculate_pointing_game(fused_up, targets_seg)
            total_saliency_iou += calculate_binary_iou(fused_pred, targets_seg).item()

            num_batches += 1

    # Summarize Standard Metrics
    acc = accuracy_score(all_targets_cls, all_preds_cls)
    f1 = f1_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    prec = precision_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    rec = recall_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    
    iou = total_iou / num_batches
    dice = total_dice / num_batches
    pix_acc = total_pix_acc / num_batches
    point_game = total_point_game / num_batches
    sal_iou = total_saliency_iou / num_batches

    print("\n=== RESULTS ===")
    print("Classification:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    
    print("\nSegmentation:")
    print(f"  IoU      : {iou:.4f}")
    print(f"  Dice     : {dice:.4f}")
    print(f"  Pixel Acc: {pix_acc:.4f}")

    print("\nInterpretability (Fast):")
    print(f"  Pointing Game : {point_game:.4f}")
    print(f"  Saliency IoU  : {sal_iou:.4f}")

    # XAI Heavy Metrics
    if num_xai_samples > 0:
        print(f"\n--- Running Heavy XAI Metrics (Insertion & Deletion) on {num_xai_samples} samples ---")
        xai_loader = DataLoader(
            CrackDataset(df, img_size=IMAGE_SIZE, augment=False),
            batch_size=1, # Must be 1 for step-by-step perturbation
            shuffle=True, # Random subset
            num_workers=0
        )
        
        del_aucs = []
        ins_aucs = []
        samples_done = 0

        for images, targets_cls, targets_seg in tqdm(xai_loader, total=num_xai_samples, desc="XAI Heavy"):
            if samples_done >= num_xai_samples:
                break
                
            images = images.to(DEVICE)
            targets_cls = targets_cls.to(DEVICE)
            
            with torch.no_grad():
                _, _, fused, _ = model(images)
                
            # Up-sample saliency map for perturbation
            fused_up = torch.nn.functional.interpolate(fused, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True)
            
            target_c = targets_cls[0].item()
            
            # Deletion
            del_auc, _ = calculate_insertion_deletion_score(model, images, fused_up, target_c, steps=10, mode='deletion')
            # Insertion
            ins_auc, _ = calculate_insertion_deletion_score(model, images, fused_up, target_c, steps=10, mode='insertion')
            
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)
            samples_done += 1
            
        print("\nInterpretability (Heavy):")
        print(f"  Deletion Score (AUC) : {np.mean(del_aucs):.4f} (LOWER is better)")
        print(f"  Insertion Score (AUC): {np.mean(ins_aucs):.4f} (HIGHER is better)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MAAL model from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file (.pt)')
    parser.add_argument('--model-module', type=str, default='src.models.maal', help='Module to import MultiTaskNetwork from (e.g. src.models.maal_expE)')
    parser.add_argument('--num-xai-samples', type=int, default=50, help='Number of samples to run Insertion/Deletion scores (default: 50). Set to 0 to skip.')
    
    args = parser.parse_args()
    evaluate(args.checkpoint, args.model_module, args.num_xai_samples)
