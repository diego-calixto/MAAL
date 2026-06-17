# Experiment Metrics Comparison

This document summarizes the validation metrics extracted from the best checkpoints of each fold across all 6 experiments stored under [resultados_cluster](file:///media/diego/HD/projects/DL_project/resultados_cluster).

## Aggregated Metrics (Mean ± Standard Deviation)

The table below shows the aggregated validation metrics across all 5 folds for each experiment, sorted by Mean Validation IoU (descending order).

| Experiment | Validation Accuracy | Validation IoU | Validation F1 | Validation Dice |
| :--- | :---: | :---: | :---: | :---: |
| **Attention** | 0.9842 ± 0.0034 | **0.8570 ± 0.0129** | 0.9886 ± 0.0023 | N/A |
| **baseline** | 0.9846 ± 0.0041 | **0.8550 ± 0.0094** | 0.9886 ± 0.0029 | 0.9032 ± 0.0076 |
| **Fusion_CAM** | 0.9852 ± 0.0043 | **0.8550 ± 0.0101** | 0.9890 ± 0.0029 | 0.9028 ± 0.0083 |
| **CAM** | 0.9860 ± 0.0039 | **0.8546 ± 0.0138** | 0.9894 ± 0.0029 | 0.9028 ± 0.0117 |
| **MAAL** | 0.9854 ± 0.0037 | **0.8512 ± 0.0138** | 0.9903 ± 0.0030 | N/A |
| **MAAL_V2** | 0.9856 ± 0.0052 | **0.8482 ± 0.0185** | 0.9903 ± 0.0030 | N/A |

> [!NOTE]
> * **Attention** does not log validation Dice score in the job output.
> * **MAAL** and **MAAL_V2** do not log validation F1 and Dice scores in the job output.

## Detailed Metrics fold-by-fold

Here are the individual metrics extracted for the best checkpoints of each fold in each experiment.

| Experiment | Fold | Best Epoch | Val Accuracy | Val IoU | Val F1 | Val Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Attention | Fold 1 | Epoch 28 | 0.9820 | 0.8620 | 0.9870 | N/A |
| Attention | Fold 2 | Epoch 29 | 0.9890 | 0.8700 | 0.9920 | N/A |
| Attention | Fold 3 | Epoch 28 | 0.9850 | 0.8380 | 0.9890 | N/A |
| Attention | Fold 4 | Epoch 28 | 0.9800 | 0.8500 | 0.9860 | N/A |
| Attention | Fold 5 | Epoch 28 | 0.9850 | 0.8650 | 0.9890 | N/A |
| CAM | Fold 1 | Epoch 16 | 0.9860 | 0.8490 | 0.9890 | 0.8990 |
| CAM | Fold 2 | Epoch 28 | 0.9920 | 0.8690 | 0.9940 | 0.9170 |
| CAM | Fold 3 | Epoch 30 | 0.9810 | 0.8380 | 0.9860 | 0.8900 |
| CAM | Fold 4 | Epoch 29 | 0.9850 | 0.8480 | 0.9890 | 0.8950 |
| CAM | Fold 5 | Epoch 28 | 0.9860 | 0.8690 | 0.9890 | 0.9130 |
| Fusion_CAM | Fold 1 | Epoch 27 | 0.9880 | 0.8570 | 0.9910 | 0.9050 |
| Fusion_CAM | Fold 2 | Epoch 27 | 0.9830 | 0.8600 | 0.9870 | 0.9070 |
| Fusion_CAM | Fold 3 | Epoch 30 | 0.9840 | 0.8430 | 0.9880 | 0.8920 |
| Fusion_CAM | Fold 4 | Epoch 30 | 0.9910 | 0.8680 | 0.9930 | 0.9130 |
| Fusion_CAM | Fold 5 | Epoch 25 | 0.9800 | 0.8470 | 0.9860 | 0.8970 |
| MAAL | Fold 1 | Epoch 23 | 0.9860 | 0.8540 | 0.9912 | N/A |
| MAAL | Fold 2 | Epoch 30 | 0.9860 | 0.8620 | 0.9892 | N/A |
| MAAL | Fold 3 | Epoch 26 | 0.9820 | 0.8340 | 0.9890 | N/A |
| MAAL | Fold 4 | Epoch 30 | 0.9910 | 0.8660 | 0.9949 | N/A |
| MAAL | Fold 5 | Epoch 21 | 0.9820 | 0.8400 | 0.9871 | N/A |
| MAAL_V2 | Fold 1 | Epoch 23 | 0.9860 | 0.8500 | 0.9912 | N/A |
| MAAL_V2 | Fold 2 | Epoch 30 | 0.9920 | 0.8670 | 0.9892 | N/A |
| MAAL_V2 | Fold 3 | Epoch 26 | 0.9840 | 0.8370 | 0.9890 | N/A |
| MAAL_V2 | Fold 4 | Epoch 30 | 0.9780 | 0.8230 | 0.9949 | N/A |
| MAAL_V2 | Fold 5 | Epoch 21 | 0.9880 | 0.8640 | 0.9871 | N/A |
| baseline | Fold 1 | Epoch 23 | 0.9880 | 0.8570 | 0.9910 | 0.9060 |
| baseline | Fold 2 | Epoch 30 | 0.9850 | 0.8600 | 0.9890 | 0.9060 |
| baseline | Fold 3 | Epoch 26 | 0.9840 | 0.8450 | 0.9880 | 0.8950 |
| baseline | Fold 4 | Epoch 30 | 0.9880 | 0.8670 | 0.9910 | 0.9130 |
| baseline | Fold 5 | Epoch 21 | 0.9780 | 0.8460 | 0.9840 | 0.8960 |