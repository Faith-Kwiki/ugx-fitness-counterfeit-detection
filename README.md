# UGX Banknote Counterfeit and Fitness Research

The current code trains a counterfeit-versus-genuine banknote classifier.
The fitness task is still planned and needs its own definition and labels.

## Set up the environment on Linux

Use Python 3.12 for the tested project setup. From the repository root:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The requirements file pins the project package versions. On the Alienware,
install or replace PyTorch with a CUDA-enabled build matched to its NVIDIA GPU.
Use the [official PyTorch installer selector](https://pytorch.org/get-started/locally/),
choose Linux, Pip, Python, and a CUDA option supported by the computer, then
run the command it provides.

Check the PyTorch build:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
~~~

For GPU training, the final value should be True.

## Dataset

The raw dataset is intentionally excluded from Git. Copy it separately under
data/raw/. It should contain one labelled_ugandan_banknote_dataset_clean
folder with splits/train_set, splits/validation_set, and splits/test_set, each
containing counterfeit and genuine subfolders. Keep the held-out test split
out of model selection.

## Optional multi-signal preprocessing cache

The RGB, edge, and LBP streams are deterministic, so they can be precomputed
once and reused. The runner below builds the cache automatically for training
and validation. Cache files go under data/processed/, which is ignored by Git.
The test split is not cached by the runner.

## Run the core ablation on the GPU desktop

From the repository root, after installing CUDA-enabled PyTorch and copying
the dataset into data/raw/:

~~~bash
bash scripts/run_core_ablation.sh
~~~

This runs four separate conditions: RGB, RGB + edges, RGB + texture, and RGB +
edges + texture. Defaults are seed 42, 15 epochs, batch size 8, and learning
rate 0.001. To override these settings, pass seed, epochs, batch size, and
learning rate in that order:

~~~bash
bash scripts/run_core_ablation.sh 43 20 16 0.0001
~~~

Each run gets a timestamped folder. The runner stops if CUDA or the required
dataset folders are missing, or if an experiment fails. It does not evaluate
the held-out test split.

The dataset, virtual environment, preprocessing cache, and checkpoint files
are not pushed to GitHub. The other desktop must receive the dataset separately.
