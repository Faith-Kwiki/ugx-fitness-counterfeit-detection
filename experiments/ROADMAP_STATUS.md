# Executable roadmap status

## Training workflows ready

- scripts/run_training_suite.sh runs the RGB CNN, ResNet-18 feature extraction, ResNet-18 fine-tuning from the matching feature-extractor checkpoint, ResNet hybrid with concatenation, and ResNet hybrid with attention.
- The suite then runs scripts/run_core_ablation.sh for RGB, RGB + edges, RGB + texture, and RGB + edges + texture using the small CNN branches.
- Each run saves its own configuration, history, validation summary, split manifest, and checkpoint. Training and model selection use train and validation data only.
- Both Linux runners perform a class-count and exact-file-hash split check before training. This cannot detect different photos of the same physical note.
- src/banknote/evaluation/evaluate_checkpoint.py evaluates one finally selected run on the held-out test data and writes accuracy, counterfeit precision/recall/F1, balanced accuracy, MCC, ROC-AUC, and a confusion-matrix image. It requires an explicit final-selection confirmation and refuses to overwrite an existing report by default.
- src/banknote/analysis/analyze_attention.py summarizes learned modality weights on validation data.
- src/banknote/analysis/gradcam.py generates a Grad-CAM overlay and metadata for one selected image and signal. The visualization indicates image regions associated with a prediction; it does not establish causal use of a security feature.

## Condition annotations completed

- `annotations/condition_labels_template.csv` now has 1,426 unique images, each with complete `soiled`, `creased`, `torn`, and `folded` tags (`yes`, `no`, or `uncertain`). Existing image paths, authenticity labels, and split assignments are retained.
- `uncertain` means the visible evidence for that tag is unclear. It is masked during training and is not treated as a negative label.
- The annotation guide defines visible image-level conditions; these labels are not an official Bank of Uganda fitness rating and do not yet define one joint fitness class.
- Current train/validation counts show only 24 positive torn labels in training and 9 in validation. Tear-head results therefore need cautious interpretation and more reviewed examples before strong claims.

## Integrated multi-task model

- `src/banknote/models/multitask_multisignal_cnn.py` shares RGB, edge, and texture features across an authenticity head and four separate condition heads.
- `src/banknote/training/train_multitask_multisignal.py` trains the heads together, masks uncertain condition values, calculates capped positive weights from the training split, and selects checkpoints using combined validation loss. It does not load the test split for training or selection.
- Run the new workflow from the repository root with `python -m src.banknote.training.train_multitask_multisignal --cache-dir data/processed/streams_v1_224 --run-id condition-tags-seed42`.
- The initial 15-epoch seed-42 weight sweep (same 998/214 train/validation split) gave best-checkpoint authenticity accuracy of 85.98% at condition-loss weight 1.0, 88.32% at 0.5, and 89.25% at 0.25. Weight 1.0 favored the four condition heads, while 0.25 was closer to the 90.65% same-split single-head baseline. The equal task-loss balance (weight 1.0) was carried forward for repeat-seed assessment.
- Three 15-epoch runs at weight 1.0 selected their checkpoints by lowest combined validation loss. Best-checkpoint validation results at threshold 0.5 were:

  | Seed | Epoch | Authenticity accuracy | Soiled F1 | Creased F1 | Torn F1 | Folded F1 |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 42 | 13 | 0.860 | 0.522 | 0.789 | 0.727 | 0.727 |
  | 43 | 13 | 0.883 | 0.515 | 0.770 | 0.383 | 0.672 |
  | 44 | 14 | 0.907 | 0.515 | 0.735 | 0.243 | 0.699 |
  | Mean | — | 0.883 | 0.517 | 0.765 | 0.451 | 0.699 |

- Across these three seeds, mean authenticity accuracy was 0.883 (sample SD 0.023). Mean condition F1 was 0.517 for soiled, 0.765 for creased, 0.451 for torn, and 0.699 for folded. Torn F1 varied substantially (sample SD 0.249) and is based on just 9 positive validation examples; it is not a stable estimate. Every condition score remains exploratory until annotations receive human spot-review.
- All runs used the same train/validation split (998/214), and none loaded or used the held-out test split for training or model selection. Keep test evaluation for the final chosen configuration.

## Still needs a project decision or new data

- Optional W&B tracking is not enabled. Local run folders already record metrics and configuration; W&B would send selected metadata to an external service and should only be enabled deliberately.
- The four visible condition outputs are a first fitness-related signal. A single fitness class still needs an agreed definition and, if required by the research question, an authoritative rating guide and labels.
- Human-review a stratified sample of visual condition annotations, especially torn-positive examples, before treating these tags as ground truth. Then assess whether annotation revisions change the validation results. Keep the test split sealed until the model and thresholds are selected.

Attention is implemented as a candidate for comparison, not proof that it improves performance. Compare it with hybrid concatenation using the same seed and settings. Use validation results for selection; reserve test metrics for final reporting.
