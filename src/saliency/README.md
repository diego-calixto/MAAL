Run instructions
From repo root:
```
python -m src.saliency.generate_saliency \
  --checkpoint checkpoints/fusion_cam/fold_0/best.pt \
  --input-dir processed_dataset_MTL/Positive/images \
  --output-dir outputs \
  --model-type fusion_cam \
  --method gradcam \
  --target-layer encoder.layer4[-1].conv3 \
  --target-type predicted \
  --threshold 0.5 \
  --device cuda
```

For vanilla saliency:
```
python -m src.saliency.generate_saliency \
  --checkpoint checkpoints/fusion_cam/fold_0/best.pt \
  --input-dir processed_dataset_MTL/Positive/images \
  --output-dir outputs \
  --model-type fusion_cam \
  --method vanilla
```