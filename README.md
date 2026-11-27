# UGX Banknote Counterfeit and Fitness Research

The baseline suite trains a counterfeit-versus-genuine banknote classifier.
The experimental multi-task model uses the same RGB, edge, and texture signals
with one authenticity head and four independent condition heads: soiled,
creased, torn, and folded. It does not create a joint fitness/authenticity
class. These visible condition tags are not an official Bank of Uganda fitness
rating.

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

## Train the integrated authenticity and condition model

The completed `annotations/condition_labels_template.csv` file supplies four
independent yes/no condition targets. `uncertain` is treated as unknown for
that condition and excluded from its loss; it is never changed to `no`. The
model shares the RGB, edge, and texture features, then predicts authenticity
and the four conditions through separate output heads. Training and model
selection use only train and validation data.

From the repository root, run:

~~~powershell
.\.venv\Scripts\python.exe -m src.banknote.training.train_multitask_multisignal --cache-dir data/processed/streams_v1_224 --run-id condition-tags-seed42
~~~

Each run saves its configuration, training history, validation summary, split
manifest, and best checkpoint under `runs/multitask/`. Validation output
reports per-condition precision, recall, F1, and average precision. Positive
class weights are calculated from the training split and capped at 20 by
default. The tear tag is sparse in the current data, so its results are
exploratory and should not be treated as a reliable deployment estimate.
Review validation results before selecting a checkpoint. Do not use the held-
out test split to choose thresholds or models.

An initial seed-42 sweep varied `--condition-loss-weight` across 1.0, 0.5,
and 0.25. Best-checkpoint validation authenticity accuracy was 85.98%, 88.32%,
and 89.25%, respectively; the same-split single-head baseline reached 90.65%.
The equal task-loss weight of 1.0 was repeated across seeds 42, 43, and 44
(998 training and 214 validation images each). Best-checkpoint validation
results at threshold 0.5 were:

| Seed | Authenticity accuracy | Soiled F1 | Creased F1 | Torn F1 | Folded F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.860 | 0.522 | 0.789 | 0.727 | 0.727 |
| 43 | 0.883 | 0.515 | 0.770 | 0.383 | 0.672 |
| 44 | 0.907 | 0.515 | 0.735 | 0.243 | 0.699 |
| Mean | 0.883 | 0.517 | 0.765 | 0.451 | 0.699 |

The torn result varied across seeds and only nine validation images were
positive for torn, so it is not a stable estimate. Condition scores remain
exploratory until the visual labels receive human spot-review. The held-out
test split was not used for training or model selection, and no test evaluation
has been run for this model.

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
