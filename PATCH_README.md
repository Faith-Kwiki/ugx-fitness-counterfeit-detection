# Phase 1 multi-signal training patch

This patch adds a reusable three-stream data loader and a standalone trainer.
It does not replace or alter any existing project file.

## Install into the project

Copy the `src/banknote/ingest/multistream_data_loader.py` and
`src/banknote/training/train_multisignal.py` files into the matching folders in
the project. Keep the existing `src/banknote/models/multisignal_cnn.py` and
preprocessing files.

## Run from the project root

```text
python -m src.banknote.training.train_multisignal --run-id concat-seed42
```

By default, the script locates the labelled dataset under `data/raw/`, uses
seed 42, trains for 15 epochs, and writes to a new folder under
`runs/multisignal/`. Each run folder contains:

- `config.json`: the run settings and environment version
- `split_manifest.csv`: split membership, labels, and file SHA-256 checksums
- `history.csv`: one row appended and flushed after each completed epoch
- `best_checkpoint.pth`: model weights from the lowest validation loss
- `latest_checkpoint.pth`: weights from the latest completed epoch
- `summary.json`: run status and best checkpoint details

The model never loads or evaluates the test images during training. The
manifest records split membership and hashes each image file so later ablations
can confirm they used the same files. Hashing reads file bytes only; it does
not send the images to the model or use them for training or model selection.
Use a different run ID for every run; an existing run folder is never
overwritten.

An example with an explicit dataset location and another seed:

```text
python -m src.banknote.training.train_multisignal --dataset-dir "data/raw/Ugandan banknote image dataset for genuine and cou/labelled_ugandan_banknote_dataset_clean" --run-id concat-seed123 --seed 123
```

Training has not been run as part of preparing this patch because the ZIP did
not include the dataset. The included source should be reviewed and tried in
the Colab project before using it for research results.
