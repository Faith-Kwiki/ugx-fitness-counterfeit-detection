#!/usr/bin/env bash
set -euo pipefail

seed=42
epochs=15
batch_size=8
learning_rate=0.001
if [ "$#" -ge 1 ]; then seed="$1"; fi
if [ "$#" -ge 2 ]; then epochs="$2"; fi
if [ "$#" -ge 3 ]; then batch_size="$3"; fi
if [ "$#" -ge 4 ]; then learning_rate="$4"; fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

python_exe="$project_root/.venv/bin/python"
if [ ! -x "$python_exe" ]; then
    echo "Missing .venv/bin/python. Create the environment from the repository root first." >&2
    exit 1
fi

cuda_available="$("$python_exe" -c 'import torch; print(torch.cuda.is_available())')"
if [ "$cuda_available" != "True" ]; then
    echo "CUDA is not available. Install a CUDA-enabled PyTorch build before training." >&2
    exit 1
fi

dataset_count="$(find data/raw -mindepth 2 -maxdepth 2 -type d -name labelled_ugandan_banknote_dataset_clean -print | wc -l)"
if [ "$dataset_count" -ne 1 ]; then
    echo "Expected exactly one labelled dataset folder two levels under data/raw; found $dataset_count." >&2
    exit 1
fi
dataset_dir="$(find data/raw -mindepth 2 -maxdepth 2 -type d -name labelled_ugandan_banknote_dataset_clean -print -quit)"
"$python_exe" -m src.banknote.validation.validate_splits --dataset-dir "$dataset_dir"
for split in train_set validation_set test_set; do
    for class_name in counterfeit genuine; do
        class_path="$dataset_dir/splits/$split/$class_name"
        if [ ! -d "$class_path" ]; then
            echo "Required dataset folder is missing: $class_path" >&2
            exit 1
        fi
    done
done

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="runs/multisignal"
for experiment in ablation-rgb ablation-rgb-edges ablation-rgb-texture ablation-rgb-edges-texture; do
    run_id="$experiment"'-seed'"$seed"'-'"$run_stamp"
    if [ -e "$run_root/$run_id" ]; then
        echo "Run folder already exists: $run_root/$run_id" >&2
        exit 1
    fi
done

echo "CUDA device:"
"$python_exe" -c 'import torch; print(torch.cuda.get_device_name(0))'

echo "Preparing train and validation streams. The test split is not cached."
"$python_exe" -m src.banknote.preprocessing.cache_streams     --dataset-dir "$dataset_dir"     --cache-dir data/processed/streams_v1_224

run_one() {
    experiment="$1"
    shift
    run_id="$experiment"'-seed'"$seed"'-'"$run_stamp"
    echo
    echo "Starting $run_id with signals: $*"
    "$python_exe" -m src.banknote.training.train_multisignal         --dataset-dir "$dataset_dir"         --cache-dir data/processed/streams_v1_224         --run-id "$run_id"         --seed "$seed"         --epochs "$epochs"         --batch-size "$batch_size"         --learning-rate "$learning_rate"         --signals "$@"
}

run_one ablation-rgb rgb
run_one ablation-rgb-edges rgb edges
run_one ablation-rgb-texture rgb texture
run_one ablation-rgb-edges-texture rgb edges texture

echo
echo "Core concatenation ablation complete. Check runs/multisignal for each run's artifacts."
echo "No test-set scores were calculated."
