# UGX Banknote Counterfeit and Fitness Research

The current code trains a counterfeit-versus-genuine banknote classifier.
The fitness task is still planned and needs its own definition and labels.

## Set up the environment on Windows

Use Python 3.12 for the tested project setup. From the repository root:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The requirements file pins the project package versions. On the GPU desktop,
install or replace PyTorch with a CUDA-enabled build matched to that computer:

- **NVIDIA GPU:** open the [official PyTorch installer selector](https://pytorch.org/get-started/locally/), choose Windows, Pip, Python, and a CUDA option supported by that computer, then run the command it provides.
- **CPU only:** the regular requirements installation is sufficient.

Check that the selected PyTorch build sees the GPU:

~~~powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
~~~

For the training desktop, torch.cuda.is_available() should print True.

## Dataset

The raw dataset is intentionally excluded from Git. Copy it separately to:

~~~text
data/raw/Ugandan banknote image dataset for genuine and cou/labelled_ugandan_banknote_dataset_clean/
~~~

It should contain splits/train_set, splits/validation_set, and splits/test_set, each with counterfeit and genuine subfolders. Keep the held-out test split out of model selection.

## Optional multi-signal preprocessing cache

The RGB, edge, and LBP streams are deterministic, so they can be precomputed once and reused. From the repository root:

~~~powershell
python -m src.banknote.preprocessing.cache_streams --dataset-dir "data/raw/Ugandan banknote image dataset for genuine and cou/labelled_ugandan_banknote_dataset_clean"
~~~

By default, this prepares training and validation only. Cache files are written under data/processed/, which is ignored by Git. The test split remains untouched unless explicitly requested.

## Start a training run

Use a new run ID for each experiment:

~~~powershell
python -m src.banknote.training.train_multisignal --cache-dir "data/processed/streams_v1_224" --run-id "concat-cache-seed42-run01" --seed 42 --epochs 15 --batch-size 8 --learning-rate 0.001
~~~

The run uses training and validation data only. It saves its configuration, history, split manifest, and checkpoints in runs/multisignal/<run-id>/. The existing trainer refuses to overwrite a run folder with the same ID.

The dataset, virtual environment, preprocessing cache, and checkpoint files are not pushed to GitHub. The other desktop must receive the dataset separately.
