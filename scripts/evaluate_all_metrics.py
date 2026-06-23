#!/usr/bin/env python3
"""
scripts/evaluate_all_metrics.py
================================
Calculates standard validation metrics and XAI metrics (Pointing Game, Saliency IoU,
Deletion AUC, Insertion AUC) for all folds of each experiment stored in resultados_cluster.

Usage:
    python scripts/evaluate_all_metrics.py --device cuda
    python scripts/evaluate_all_metrics.py --num-xai-samples 50 --device cpu
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.common import (
    IMAGE_SIZE,
    CrackDataset,
    prepare_dataframe,
    calculate_iou,
    calculate_dice,
    calculate_pixel_accuracy,
    calculate_pointing_game,
    calculate_binary_iou,
    calculate_insertion_deletion_score,
)
from src.saliency.utils import load_model_from_checkpoint, resolve_target_layer
from src.saliency.methods import GradCAM

# Experiment configurations matching visualize_fold0.py
EXPERIMENT_CONFIGS = {
    "baseline": {
        "model_type": "baseline",
        "checkpoint_pattern": "baseline/checkpoints/fold_{fold}/best.pt"
    },
    "Attention": {
        "model_type": "attention",
        "checkpoint_pattern": "attention/checkpoints/fold_{fold}/best.pt"
    },
    "CAM": {
        "model_type": "cam_head",
        "checkpoint_pattern": "cam/checkpoints/fold_{fold}/best.pt"
    },
    "Fusion_CAM": {
        "model_type": "fusion_cam",
        "checkpoint_pattern": "fusion_cam/checkpoints/fold_{fold}/best.pt"
    },
    "MAAL": {
        "model_type": "maal",
        "checkpoint_pattern": "maal/checkpoints/fold_{fold}/best.pt"
    },
    "MAAL_V2": {
        "model_type": "maal",
        "checkpoint_pattern": "maal_v2/checkpoints/fold_{fold}/best.pt"
    },
    "MAAL_V3": {
        "model_type": "maal_expg",
        "checkpoint_pattern": "maal_v3/checkpoints/fold_{fold}/best.pt"
    },
    "MAAL_V4": {
        "model_type": "maal_exph",
        "checkpoint_pattern": "maal_v4/checkpoints/fold_{fold}/best.pt"
    }
}

TARGET_LAYER_NAME = "encoder.layer4[-1].conv3"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all metrics on resultados_cluster checkpoints.")
    parser.add_argument(
        "--dataset-dir", type=str, default="processed_dataset_MTL",
        help="Path to the dataset directory (default: processed_dataset_MTL)."
    )
    parser.add_argument(
        "--resultados-dir", type=str, default="resultados_cluster",
        help="Path to the results directory (default: resultados_cluster)."
    )
    parser.add_argument(
        "--num-xai-samples", type=int, default=100,
        help="Number of images per fold to evaluate XAI metrics (Grad-CAM, Pointing Game, Deletion, Insertion). "
             "Use -1 to run on the entire validation fold (warning: slow on CPU). Default: 100."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run evaluation on (e.g. cuda, cpu). Default: cuda if available."
    )
    parser.add_argument(
        "--sort-df", type=str, default="none", choices=["none", "filename", "path"],
        help="How to sort the dataset dataframe before cross-validation splitting. Default: none."
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Subset of experiments to evaluate. E.g. --experiments baseline MAAL"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run evaluation on only 2 images per fold to verify execution quickly."
    )
    return parser.parse_args()


def get_dataset_splits(dataset_dir, sort_df="none"):
    df = prepare_dataframe(dataset_dir)
    if len(df) == 0:
        raise ValueError(f"Dataset in '{dataset_dir}' is empty or not found.")
        
    print(f"Loaded dataset with {len(df)} images.")
    
    if sort_df == "filename":
        print("Sorting DataFrame by filename for splitting...")
        df = df.sort_values(by="filename").reset_index(drop=True)
    elif sort_df == "path":
        print("Sorting DataFrame by path for splitting...")
        df = df.sort_values(by="path").reset_index(drop=True)
        
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, df['label'], df['group']))
    return df, splits


def evaluate_checkpoint(
    model, 
    val_loader, 
    device, 
    num_xai_samples=100, 
    dry_run=False
):
    model.eval()
    
    # 1. Evaluate standard metrics on the full validation fold
    all_preds_cls = []
    all_targets_cls = []
    total_iou = 0.0
    total_dice = 0.0
    total_pix_acc = 0.0
    num_batches = 0
    
    print("  -> Evaluating standard metrics (full validation fold)...")
    with torch.no_grad():
        for images, targets_cls, targets_seg in val_loader:
            images = images.to(device)
            targets_cls = targets_cls.to(device)
            targets_seg = targets_seg.to(device)
            
            outputs = model(images)
            y_cls, y_seg = outputs[0], outputs[1]
            
            # Classification
            preds_cls = torch.argmax(y_cls, dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())
            
            # Segmentation
            total_iou += calculate_iou(y_seg, targets_seg).item()
            total_dice += calculate_dice(y_seg, targets_seg, threshold=0.5, from_logits=True).item()
            total_pix_acc += calculate_pixel_accuracy(y_seg, targets_seg).item()
            
            num_batches += 1
            if dry_run and num_batches >= 1:
                break
                
    acc = accuracy_score(all_targets_cls, all_preds_cls)
    f1 = f1_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    prec = precision_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    rec = recall_score(all_targets_cls, all_preds_cls, average='binary', zero_division=0)
    
    iou = total_iou / num_batches
    dice = total_dice / num_batches
    pix_acc = total_pix_acc / num_batches
    
    # 2. Evaluate saliency and XAI metrics on a subset
    point_game_score = 0.0
    saliency_iou_score = 0.0
    deletion_auc = 0.0
    insertion_auc = 0.0
    
    if num_xai_samples == 0:
        return acc, f1, prec, rec, iou, dice, pix_acc, 0.0, 0.0, 0.0, 0.0

    print(f"  -> Evaluating XAI metrics (Grad-CAM) on subset...")
    # Set up Grad-CAM explainer
    target_layer = resolve_target_layer(model, TARGET_LAYER_NAME)
    explainer = GradCAM(
        model=model,
        target_layer=target_layer,
        device=device,
        target_type="soft",
        threshold=0.5,
    )
    
    # Re-sample validation dataset for XAI (single batch execution)
    val_dataset = val_loader.dataset
    indices = list(range(len(val_dataset)))
    
    # Deterministic subset sampling per fold
    np.random.seed(42)
    np.random.shuffle(indices)
    
    if dry_run:
        subset_indices = indices[:2]
    elif num_xai_samples > 0:
        subset_indices = indices[:min(num_xai_samples, len(val_dataset))]
    else:
        subset_indices = indices # full fold
        
    hits = 0.0
    num_pointing_objects = 0
    total_sal_iou = 0.0
    del_aucs = []
    ins_aucs = []
    
    for idx in tqdm(subset_indices, desc="    XAI evaluation", leave=False):
        image, target_cls, target_seg = val_dataset[idx]
        
        # Add batch dim
        image_batch = image.unsqueeze(0).to(device)
        target_seg_batch = target_seg.unsqueeze(0).to(device)
        target_cls_val = int(target_cls.item())
        
        # Generate Saliency Map
        try:
            outputs = model(image_batch)
            if len(outputs) >= 3:
                # Use intrinsic CAM map for MAAL, Fusion_CAM and CAM_head
                intrinsic_logits = outputs[2] # fused or cam_logits
                intrinsic_up = torch.nn.functional.interpolate(
                    intrinsic_logits, 
                    size=(image_batch.shape[2], image_batch.shape[3]), 
                    mode='bilinear', 
                    align_corners=True
                )
                cam_tensor = torch.sigmoid(intrinsic_up) # [1, 1, H, W]
            else:
                # Fallback to Grad-CAM for baseline and attention
                cam = explainer.generate(image_batch) # 2D numpy [H, W] normalized [0, 1]
                cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
        except Exception as e:
            warnings.warn(f"Failed to generate CAM: {e}")
            continue
            
        # Pointing Game (using custom logic to count instances with objects)
        if target_seg_batch[0, 0].max() > 0:
            num_pointing_objects += 1
            max_idx = torch.argmax(cam_tensor[0, 0]).item()
            H, W = cam_tensor.shape[2:]
            y = max_idx // W
            x = max_idx % W
            if target_seg_batch[0, 0, y, x] > 0.5:
                hits += 1.0
                
        # Saliency IoU (binarized Grad-CAM vs GT segmentations)
        pred_bin = (cam_tensor > 0.5).float()
        total_sal_iou += calculate_binary_iou(pred_bin, target_seg_batch).item()
        
        # Deletion AUC
        del_auc, _ = calculate_insertion_deletion_score(
            model, image_batch, cam_tensor, target_cls_val, steps=10, mode='deletion'
        )
        # Insertion AUC
        ins_auc, _ = calculate_insertion_deletion_score(
            model, image_batch, cam_tensor, target_cls_val, steps=10, mode='insertion'
        )
        
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
        
    point_game_score = hits / max(1, num_pointing_objects)
    saliency_iou_score = total_sal_iou / len(subset_indices)
    deletion_auc = np.mean(del_aucs) if del_aucs else 0.0
    insertion_auc = np.mean(ins_aucs) if ins_aucs else 0.0
    
    return (
        acc, f1, prec, rec, iou, dice, pix_acc,
        point_game_score, saliency_iou_score, deletion_auc, insertion_auc
    )


def df_to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Fallback to manual markdown table formatting
        headers = list(df.columns)
        widths = [len(h) for h in headers]
        rows = []
        for _, r in df.iterrows():
            row_vals = [str(val) for val in r]
            rows.append(row_vals)
            for i, val in enumerate(row_vals):
                widths[i] = max(widths[i], len(val))
                
        header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
        separator = " | ".join("-" * widths[i] for i in range(len(headers)))
        markdown_lines = ["| " + header_str + " |", "| " + separator + " |"]
        for r in rows:
            row_str = " | ".join(f"{val:<{widths[i]}}" for i, val in enumerate(r))
            markdown_lines.append("| " + row_str + " |")
        return "\n".join(markdown_lines)


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")
    
    resultados_path = Path(args.resultados_dir)
    dataset_path = Path(args.dataset_dir)
    
    # 1. Retrieve splits
    df, splits = get_dataset_splits(dataset_path, args.sort_df)
    
    # Determine experiments to evaluate
    experiments = args.experiments or list(EXPERIMENT_CONFIGS.keys())
    
    detailed_results = []
    
    for exp_name in experiments:
        if exp_name not in EXPERIMENT_CONFIGS:
            print(f"Warning: Unknown experiment '{exp_name}' – skipping.")
            continue
            
        cfg = EXPERIMENT_CONFIGS[exp_name]
        model_type = cfg["model_type"]
        checkpoint_pattern = cfg["checkpoint_pattern"]
        
        print(f"\nEvaluating Experiment: {exp_name} (Model type: {model_type})")
        
        exp_metrics = {
            "Accuracy": [], "F1-Score": [], "Precision": [], "Recall": [],
            "IoU": [], "Dice": [], "Pixel_Acc": [],
            "Pointing_Game": [], "Saliency_IoU": [], "Deletion_AUC": [], "Insertion_AUC": []
        }
        
        for fold in range(5):
            checkpoint_path = resultados_path / checkpoint_pattern.format(fold=fold)
            print(f"  Fold {fold+1}/5 | Checkpoint: {checkpoint_pattern.format(fold=fold)}")
            
            if not checkpoint_path.exists():
                print(f"    [SKIP] Checkpoint file does not exist: {checkpoint_path}")
                continue
                
            # Load validation set for this fold
            train_idx, val_idx = splits[fold]
            val_df = df.iloc[val_idx]
            
            val_loader = DataLoader(
                CrackDataset(val_df, img_size=IMAGE_SIZE, augment=False),
                batch_size=32,
                shuffle=False,
                num_workers=2
            )
            
            # Load model
            try:
                model, _ = load_model_from_checkpoint(str(checkpoint_path), model_type, device)
            except Exception as e:
                print(f"    [ERROR] Failed to load model or checkpoint: {e}")
                continue
                
            # Evaluate metrics
            t_start = time.time()
            metrics = evaluate_checkpoint(
                model, 
                val_loader, 
                device, 
                num_xai_samples=args.num_xai_samples,
                dry_run=args.dry_run
            )
            t_elapsed = time.time() - t_start
            
            # Unpack results
            acc, f1, prec, rec, iou, dice, pix_acc, pg, sal_iou, del_auc, ins_auc = metrics
            
            print(f"    Metrics: Acc={acc:.4f} IoU={iou:.4f} Dice={dice:.4f} PG={pg:.4f} Del={del_auc:.4f} Ins={ins_auc:.4f} ({t_elapsed:.1f}s)")
            
            # Store detailed row
            detailed_results.append({
                "Experiment": exp_name,
                "Fold": fold,
                "Accuracy": acc,
                "F1-Score": f1,
                "Precision": prec,
                "Recall": rec,
                "IoU": iou,
                "Dice": dice,
                "Pixel Accuracy": pix_acc,
                "Pointing Game": pg,
                "Saliency IoU": sal_iou,
                "Deletion Score (AUC)": del_auc,
                "Insertion Score (AUC)": ins_auc
            })
            
            # Accumulate for experiment summary
            exp_metrics["Accuracy"].append(acc)
            exp_metrics["F1-Score"].append(f1)
            exp_metrics["Precision"].append(prec)
            exp_metrics["Recall"].append(rec)
            exp_metrics["IoU"].append(iou)
            exp_metrics["Dice"].append(dice)
            exp_metrics["Pixel_Acc"].append(pix_acc)
            exp_metrics["Pointing_Game"].append(pg)
            exp_metrics["Saliency_IoU"].append(sal_iou)
            exp_metrics["Deletion_AUC"].append(del_auc)
            exp_metrics["Insertion_AUC"].append(ins_auc)
            
        # Log aggregated metrics for this experiment
        print(f"\n--- Aggregated results for {exp_name} (Mean value for all folds) ---")
        for k, v in exp_metrics.items():
            if v:
                print(f"  {k:20}: {np.mean(v):.4f} ± {np.std(v):.4f}")
                
    # 3. Save files
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        
        # Calculate summary/aggregated dataframe (Mean ± Std)
        summary_rows = []
        for exp_name, group in df_detailed.groupby("Experiment"):
            row = {"Experiment": exp_name}
            for col in group.columns:
                if col not in ["Experiment", "Fold"]:
                    mean_val = group[col].mean()
                    std_val = group[col].std()
                    row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
                    # Keep raw floats for mathematical sorting in final display
                    row[f"{col}_mean"] = mean_val
            summary_rows.append(row)
            
        df_summary = pd.DataFrame(summary_rows)
        # Sort by Mean IoU descending
        df_summary = df_summary.sort_values(by="IoU_mean", ascending=False).reset_index(drop=True)
        # Drop helper columns
        clean_cols = [c for c in df_summary.columns if not c.endswith("_mean")]
        df_summary_clean = df_summary[clean_cols]
        
        # Write CSV and XLSX outputs
        os.makedirs(resultados_path, exist_ok=True)
        detailed_csv_path = resultados_path / "evaluated_metrics_detailed.csv"
        summary_csv_path = resultados_path / "evaluated_metrics_summary.csv"
        detailed_xlsx_path = resultados_path / "evaluated_metrics_detailed.xlsx"
        summary_xlsx_path = resultados_path / "evaluated_metrics_summary.xlsx"
        
        df_detailed.to_csv(detailed_csv_path, index=False)
        df_summary_clean.to_csv(summary_csv_path, index=False)
        
        try:
            df_detailed.to_excel(detailed_xlsx_path, index=False)
            df_summary_clean.to_excel(summary_xlsx_path, index=False)
            print(f"\nSaved detailed fold-by-fold results to: {detailed_csv_path} and {detailed_xlsx_path}")
            print(f"Saved aggregated summaries to: {summary_csv_path} and {summary_xlsx_path}")
        except ModuleNotFoundError:
            print(f"\nSaved detailed fold-by-fold results to: {detailed_csv_path}")
            print(f"Saved aggregated summaries to: {summary_csv_path}")
            print("\n[WARNING] 'openpyxl' module not found. Could not save .xlsx files. Run 'pip install openpyxl' to fix this.")
        
        # Display as a markdown table
        print("\n=== AGGREGATED METRICS TABLE ===")
        print(df_to_markdown(df_summary_clean))
        
        # Save a copy in markdown format
        summary_md_path = resultados_path / "evaluated_metrics_summary.md"
        with open(summary_md_path, "w") as f:
            f.write("# Calculated Experiment Metrics (Aggregated)\n\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: Device={device}, Sort={args.sort_df}, XAI-Samples={args.num_xai_samples}\n\n")
            f.write(df_to_markdown(df_summary_clean))
            f.write("\n")
            
        print(f"Saved aggregated markdown report to: {summary_md_path}")
        
    else:
        print("No evaluations were run.")


if __name__ == "__main__":
    main()
