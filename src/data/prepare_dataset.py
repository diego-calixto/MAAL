import argparse
import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def extract_rar(rar_path: Path, extract_dir: Path) -> None:
    """Extract a .rar archive into the target directory."""
    if not rar_path.exists():
        raise FileNotFoundError(f"RAR file not found: {rar_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    unrar_cmd = shutil.which("unrar")
    if unrar_cmd is None:
        raise RuntimeError(
            "The 'unrar' executable is required to extract .rar archives." "\nInstall it with your package manager, e.g. 'sudo apt install unrar'."
        )

    subprocess.run([unrar_cmd, "x", "-y", str(rar_path), str(extract_dir)], check=True)
    print(f"Extraction completed: {rar_path} -> {extract_dir}")


def create_balanced_dataset(
    source_dir: Path,
    output_dir: Path,
    crop_size: int = 224,
    negative_ratio: float = 0.14,
) -> dict:
    """Build a balanced multi-task patch dataset from raw images and masks.

    The output contains two classes (Positive, Negative), each with:
    - images/   (RGB patch)
    - masks/    (corresponding binary mask patch)
    This supports MTL workflows combining classification and segmentation.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- INICIANDO PROCESSAMENTO ---")
    print(f"Fonte: {source_dir}")
    print(f"Destino: {output_dir}")
    print(f"Taxa de Negativos (Fundo): {negative_ratio * 100:.1f}%")

    all_jpgs = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.JPG"))
    image_files = []
    mask_files = []

    for p in all_jpgs:
        path_str = str(p)
        if "BW" in path_str or "bw" in path_str:
            mask_files.append(p)
        elif "rgb" in path_str or "RGB" in path_str:
            image_files.append(p)

    print(f"Mapeado: {len(image_files)} originais e {len(mask_files)} máscaras.")
    mask_map = {m.stem: m for m in mask_files}
    stats = {"Positive": 0, "Negative": 0, "Discarded": 0}

    for img_path in tqdm(image_files, desc="processing images"):
        file_id = img_path.stem
        if file_id not in mask_map:
            continue

        mask_path = mask_map[file_id]
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        h, w = img.shape[:2]

        for y in range(0, h - crop_size + 1, crop_size):
            for x in range(0, w - crop_size + 1, crop_size):
                img_crop = img[y : y + crop_size, x : x + crop_size]
                mask_crop = mask[y : y + crop_size, x : x + crop_size]
                white_pixels = np.sum(mask_crop >= 127)

                if white_pixels > 500:
                    label = "Positive"
                    should_save = True
                else:
                    label = "Negative"
                    should_save = random.random() <= negative_ratio

                if should_save:
                    save_dir_img = output_dir / label / "images"
                    save_dir_msk = output_dir / label / "masks"
                    save_dir_img.mkdir(parents=True, exist_ok=True)
                    save_dir_msk.mkdir(parents=True, exist_ok=True)

                    filename_base = f"{file_id}_{y}_{x}"
                    cv2.imwrite(str(save_dir_img / f"{filename_base}.jpg"), img_crop)
                    cv2.imwrite(str(save_dir_msk / f"{filename_base}.png"), mask_crop)
                    stats[label] += 1
                else:
                    stats["Discarded"] += 1

    print("\n--- RESUMO FINAL ---")
    print(f"Imagens Positivas (Rachadura): {stats['Positive']}")
    print(f"Imagens Negativas (Fundo):  {stats['Negative']}")
    print(f"Imagens Descartadas:        {stats['Discarded']}")
    print(f"Total Salvo:                {stats['Positive'] + stats['Negative']}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a dataset from a .rar archive and prepare a multi-task patch dataset with images and masks.")
    parser.add_argument("--rar", type=Path, default=Path("dataset.rar"), help="Path to the dataset .rar archive.")
    parser.add_argument("--extract-dir", type=Path, default=Path("raw_data"), help="Directory where the archive will be extracted.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_dataset_MTL"), help="Directory where processed MTL patches will be saved.")
    parser.add_argument("--crop-size", type=int, default=224, help="Patch crop size.")
    parser.add_argument("--negative-ratio", type=float, default=0.14, help="Fraction of negative patches to keep.")
    parser.add_argument("--no-extract", action="store_true", help="Skip .rar extraction and use the existing extract-dir content.")
    args = parser.parse_args()

    if not args.no_extract:
        print("Verificando e extraindo arquivo .rar...")
        extract_rar(args.rar, args.extract_dir)
    else:
        print("Pulando extração .rar. Usando conteúdo existente em:", args.extract_dir)

    create_balanced_dataset(
        source_dir=args.extract_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        negative_ratio=args.negative_ratio,
    )


if __name__ == "__main__":
    main()
