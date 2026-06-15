#!/usr/bin/env python3
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def gather_images(gradcams_dir, max_images=15):
	p = Path(gradcams_dir)
	exts = ('*.png', '*.jpg', '*.jpeg')
	files = []
	for e in exts:
		files.extend(list(p.rglob(e)))
	files = sorted(files)
	return files[:max_images]


def plot_grid(image_paths, rows=3, cols=5, figsize=(15, 9), save_path=None, row_labels=None):
	fig, axes = plt.subplots(rows, cols, figsize=figsize)
	axes = axes.flatten()
	for ax in axes:
		ax.axis('off')

	for i, path in enumerate(image_paths):
		try:
			img = Image.open(path).convert('RGB')
			axes[i].imshow(img)
		except Exception as e:
			axes[i].text(0.5, 0.5, f'Error\n{e}', ha='center')

	# add row labels if provided
	if row_labels is not None:
		for r, label in enumerate(row_labels[:rows]):
			ax = axes[r * cols]
			pos = ax.get_position()
			fig.text(
				0.02,
				pos.y0 + pos.height / 2,
				label,
				ha='left',
				va='center',
				fontsize=16,
				fontweight='bold',
			)

	# fill remaining axes if fewer than rows*cols
	for j in range(len(image_paths), rows * cols):
		axes[j].imshow(Image.new('RGB', (10, 10), (255, 255, 255)))

	plt.tight_layout(rect=[0.12, 0, 1, 1])
	if save_path:
		out = Path(save_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out, dpi=200)
		print(f"Saved grid to {out}")
	plt.show()


def main():
	base = Path(__file__).parent
	gradcams_dir = base / 'gradcams'
	image_paths = gather_images(gradcams_dir)
	if not image_paths:
		print(f"No images found in {gradcams_dir}")
		return
	save_path = Path(__file__).parents[2] / 'outputs' / 'gradcams_grid.png'
	plot_grid(
		image_paths,
		rows=3,
		cols=5,
		save_path=str(save_path),
		row_labels=['Attention', 'CAM', 'Fusion'],
	)


if __name__ == '__main__':
	main()

