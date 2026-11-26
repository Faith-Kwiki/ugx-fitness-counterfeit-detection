# Executable roadmap status

## Runnable now

- The multi-signal trainer writes each run to a unique folder and records its configuration, history, validation summary, split manifest, and checkpoint.
- The optional preprocessing cache stores deterministic RGB, edge, and LBP arrays for the train and validation splits.
- scripts/run_core_ablation.ps1 checks CUDA, the dataset, and run folders, builds the cache, then runs four seed-matched conditions:
  1. RGB
  2. RGB + edges
  3. RGB + texture
  4. RGB + edges + texture
- All four conditions use concatenation and the same training settings. The runner does not evaluate the test split.

## Still needs implementation before it can be run as a roadmap phase

- Reproducible, run-isolated RGB CNN and ResNet-18 feature-extraction/fine-tuning trainers. The older scripts use fixed output filenames and are not included in the runner.
- ResNet-18 RGB plus edge/texture hybrid model.
- Attention-based fusion.
- A final test-set evaluator with the planned classification metrics and confusion matrix.
- Grad-CAM and structured error analysis.
- A defined fitness label/task and data source.

The existing RGB, edge, and texture concatenation model is the current implemented multi-signal architecture. Attention and the ResNet hybrid are not implemented by this runner.
