# MAAL

Deep learning utilities and experiments for the MAAL project.

Overview
-
This repository contains code, notebooks and utilities for training and evaluating deep learning models related to the MAAL project. It includes model definitions, data preparation scripts and example notebooks for running experiments locally or in Colab.

Repository structure
-
- `MAAL.py` - Main training / experiment script for the MAAL model.
- `Attention.py` - Attention mechanism implementations and helpers.
- `common.py` - Shared utilities and helper functions used across scripts.
- `prepare_dataset.py` - Scripts to prepare and preprocess the dataset used by experiments.
- `Colab_Run.ipynb` - Notebook with an example Colab-compatible run.
- `V2_multi_task_fissuras.ipynb` - Notebook for multi-task experiments and analysis.

Requirements
-
- Python 3.8 or newer
- Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

- Install dependencies (if a `requirements.txt` exists, otherwise install your project's deps):

```bash
pip install -r requirements.txt
```

Quick start
-
1. Prepare the data:

```bash
python prepare_dataset.py
```

2. Run training / experiments (example):

```bash
python MAAL.py --config configs/your_config.yaml
```

3. Open and run the notebooks for exploratory experiments:

- Use `Colab_Run.ipynb` to run in Google Colab.
- Use `V2_multi_task_fissuras.ipynb` for multi-task experiments and visualization.

Notes
-
- If you maintain a `requirements.txt`, add it to the repo so others can reproduce your environment.
- Adjust configuration flags or add a `configs/` folder for reproducible experiment settings.
- `prepare_dataset.py` includes dataset-specific preprocessing — review it before running if you have local data organization differences.

Contact
-
For questions about the code or experiments, open an issue or contact the repository owner.

