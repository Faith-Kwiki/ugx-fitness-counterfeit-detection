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
    echo "Missing .venv/bin/python. Set up the project environment first." >&2
    exit 1
fi
if [ "$("$python_exe" -c 'import torch; print(torch.cuda.is_available())')" != "True" ]; then
    echo "CUDA is not available. Install a CUDA-enabled PyTorch build first." >&2
    exit 1
fi

dataset_count="$(find data/raw -mindepth 2 -maxdepth 2 -type d -name labelled_ugandan_banknote_dataset_clean -print | wc -l)"
if [ "$dataset_count" -ne 1 ]; then
    echo "Expected exactly one labelled dataset folder under data/raw; found $dataset_count." >&2
    exit 1
fi
dataset_dir="$(find data/raw -mindepth 2 -maxdepth 2 -type d -name labelled_ugandan_banknote_dataset_clean -print -quit)"
"$python_exe" -m src.banknote.validation.validate_splits --dataset-dir "$dataset_dir"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="runs/research"
for model_name in rgb_cnn resnet18_frozen resnet18_finetune hybrid_concat hybrid_attention; do
    run_id="suite-$model_name-seed$seed-$run_stamp"
    if [ -e "$run_root/$run_id" ]; then
        echo "Run folder already exists: $run_root/$run_id" >&2
        exit 1
    fi
done

"$python_exe" -m src.banknote.preprocessing.cache_streams     --dataset-dir "$dataset_dir"     --cache-dir data/processed/streams_v1_224

run_model() {
    model_name="$1"
    run_id="$2"
    shift 2
    "$python_exe" -m src.banknote.training.train_research_model         --model "$model_name"         --dataset-dir "$dataset_dir"         --cache-dir data/processed/streams_v1_224         --run-id "$run_id"         --seed "$seed"         --epochs "$epochs"         --batch-size "$batch_size"         --learning-rate "$learning_rate"         "$@"
}

rgb_id="suite-rgb_cnn-seed$seed-$run_stamp"
frozen_id="suite-resnet18_frozen-seed$seed-$run_stamp"
finetune_id="suite-resnet18_finetune-seed$seed-$run_stamp"
concat_id="suite-hybrid_concat-seed$seed-$run_stamp"
attention_id="suite-hybrid_attention-seed$seed-$run_stamp"

run_model rgb_cnn "$rgb_id"
run_model resnet18_frozen "$frozen_id"
run_model resnet18_finetune "$finetune_id"     --learning-rate 0.0001     --backbone-learning-rate 0.00001     --init-checkpoint "$run_root/$frozen_id/best_checkpoint.pth"
run_model hybrid_concat "$concat_id"
run_model hybrid_attention "$attention_id"

bash scripts/run_core_ablation.sh "$seed" "$epochs" "$batch_size" "$learning_rate"

echo
echo "Training suite complete. No test evaluation was run."
echo "Review validation results and select models before running the test evaluator."
