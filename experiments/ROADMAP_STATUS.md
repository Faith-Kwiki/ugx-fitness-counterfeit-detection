# Executable roadmap status

## Training workflows ready

- scripts/run_training_suite.sh runs the RGB CNN, ResNet-18 feature extraction, ResNet-18 fine-tuning from the matching feature-extractor checkpoint, ResNet hybrid with concatenation, and ResNet hybrid with attention.
- The suite then runs scripts/run_core_ablation.sh for RGB, RGB + edges, RGB + texture, and RGB + edges + texture using the small CNN branches.
- Each run saves its own configuration, history, validation summary, split manifest, and checkpoint. Training and model selection use train and validation data only.
- Both Linux runners perform a class-count and exact-file-hash split check before training. This cannot detect different photos of the same physical note.
- src/banknote/evaluation/evaluate_checkpoint.py evaluates one finally selected run on the held-out test data and writes accuracy, counterfeit precision/recall/F1, balanced accuracy, MCC, ROC-AUC, and a confusion-matrix image. It requires an explicit final-selection confirmation and refuses to overwrite an existing report by default.
- src/banknote/analysis/analyze_attention.py summarizes learned modality weights on validation data.
- src/banknote/analysis/gradcam.py generates a Grad-CAM overlay and metadata for one selected image and signal. The visualization indicates image regions associated with a prediction; it does not establish causal use of a security feature.

## Still needs a project decision or new data

- Optional W&B tracking is not enabled. Local run folders already record metrics and configuration; W&B would send selected metadata to an external service and should only be enabled deliberately.
- The fitness task needs a precise label definition, an annotation process, and a fitness-labeled dataset. The current repository only has counterfeit-versus-genuine labels.

Attention is implemented as a candidate for comparison, not proof that it improves performance. Compare it with hybrid concatenation using the same seed and settings. Use validation results for selection; reserve test metrics for final reporting.
