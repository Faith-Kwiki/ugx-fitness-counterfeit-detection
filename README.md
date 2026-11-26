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

The raw dataset is intentionally excluded from Git. Copy it so the labelled
folder is two levels below data/raw/, for example
data/raw/source-folder/labelled_ugandan_banknote_dataset_clean/. It should
contain splits/train_set, splits/validation_set, and splits/test_set, each
containing counterfeit and genuine subfolders. Keep the held-out test split
out of model selection.

## Optional multi-signal preprocessing cache

The RGB, edge, and LBP streams are deterministic, so they can be precomputed
once and reused. The runner below builds the cache automatically for training
and validation. Cache files go under data/processed/, which is ignored by Git.
The test split is not cached by the runner.

## Run the training suite on the GPU desktop

After installing CUDA-enabled PyTorch and copying the dataset into data/raw/,
run from the repository root:

~~~bash
bash scripts/run_training_suite.sh
~~~

The first ResNet run may download pretrained ImageNet weights, so the GPU
computer needs internet access for that initial download. The suite trains
the RGB CNN; ResNet-18 frozen and fine-tuned baselines; the
ResNet hybrid with concatenation and attention; then RGB, RGB + edges,
RGB + texture, and RGB + edges + texture ablations. Defaults are seed 42,
15 epochs, batch size 8, and learning rate 0.001. To override them, pass seed,
epochs, batch size, and learning rate in that order:

~~~bash
bash scripts/run_training_suite.sh 43 20 16 0.0001
~~~

Before training, the runner reports class counts and stops if byte-identical
images occur in more than one split. This check does not identify different
photos of the same physical note.

This is a long suite. It checks CUDA and the dataset before starting, uses
separate run folders, and does not evaluate the test split.

## Final test evaluation

After choosing final models from validation results, prepare the test cache
and evaluate each selected run on the held-out test set:

~~~bash
python -m src.banknote.preprocessing.cache_streams --dataset-dir data/raw/source-folder/labelled_ugandan_banknote_dataset_clean --splits test_set
python -m src.banknote.evaluation.evaluate_checkpoint --run-dir runs/research/SELECTED_RUN_ID --cache-dir data/processed/streams_v1_224 --confirm-final-selection
~~~

The evaluator only accepts a completed run, requires both classes in the
test split, and refuses to overwrite an existing report unless --allow-rerun
is explicitly supplied. It writes accuracy, counterfeit precision/recall/F1,
counterfeit false-acceptance rate, genuine false-rejection rate, balanced
accuracy, MCC, ROC-AUC, and a confusion-matrix image into the run folder.


## Analyze hybrid attention weights

For a completed hybrid-attention run, summarize learned RGB, edge, and texture weights on validation data:

~~~bash
python -m src.banknote.analysis.analyze_attention --run-dir runs/research/SELECTED_ATTENTION_RUN --cache-dir data/processed/streams_v1_224
~~~

It writes a JSON summary and a plot, including the mean weights across validation images and by true validation class. This is a diagnostic, not evidence that the model relies on a particular security feature.

## Grad-CAM visualization

After training, use a completed run and one image to save a Grad-CAM overlay:

~~~bash
python -m src.banknote.analysis.gradcam --run-dir runs/research/SELECTED_RUN_ID --image "path/to/banknote.jpg" --signal rgb
~~~

For multi-signal runs, signal can be rgb, edges, or texture if that signal is in the model. To explain the genuine class instead of the predicted class, add --class-index 1; counterfeit is class 0. The overlay and a JSON sidecar are saved next to the run artifacts. Grad-CAM shows regions associated with a prediction, not proof that a security feature caused it.

The dataset, virtual environment, preprocessing cache, and checkpoint files
are not pushed to GitHub. The other desktop must receive the dataset separately.
