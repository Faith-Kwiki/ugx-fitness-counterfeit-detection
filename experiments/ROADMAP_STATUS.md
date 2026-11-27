# Executable roadmap status

## Training workflows ready

- scripts/run_training_suite.sh runs the RGB CNN, ResNet-18 feature extraction, ResNet-18 fine-tuning from the matching feature-extractor checkpoint, ResNet hybrid with concatenation, and ResNet hybrid with attention.
- The suite then runs scripts/run_core_ablation.sh for RGB, RGB + edges, RGB + texture, and RGB + edges + texture using the small CNN branches.
- Each run saves its own configuration, history, validation summary, split manifest, and checkpoint. Training and model selection use train and validation data only.
- Both Linux runners perform a class-count and exact-file-hash split check before training. This cannot detect different photos of the same physical note.
- `src/banknote/evaluation/evaluate_checkpoint.py` evaluates a completed multi-task checkpoint on validation or held-out test data. Test evaluation requires explicit final-selection confirmation, checks that both authenticity classes are present, and refuses to overwrite an existing report by default. It records counterfeit metrics, all four condition-head metrics, and a test confusion-matrix image.
- src/banknote/analysis/analyze_attention.py summarizes learned modality weights on validation data.
- src/banknote/analysis/gradcam.py generates a Grad-CAM overlay and metadata for one selected image and signal. The visualization indicates image regions associated with a prediction; it does not establish causal use of a security feature.

## Condition annotations completed

- `annotations/condition_labels_template.csv` now has 1,426 unique images, each with complete `soiled`, `creased`, `torn`, and `folded` tags (`yes`, `no`, or `uncertain`). Existing image paths, authenticity labels, and split assignments are retained.
- `uncertain` means the visible evidence for that tag is unclear. It is masked during training and is not treated as a negative label.
- The annotation guide defines visible image-level conditions; these labels are not an official Bank of Uganda fitness rating and do not yet define one joint fitness class.
- Current train/validation counts show 24 positive torn labels in training and 12 known positives in validation after spot-review. Tear-head results need cautious interpretation and more reviewed examples before strong claims.
- The 39-image blinded validation spot-review is complete. All 156 condition decisions have `yes`, `no`, or `uncertain` values, with reviewer confidence captured. The blind-review CSV now contains the submitted labels and comments; `annotations/condition_spot_review_disagreements.csv` records the 58 differences from the previous labels: 24 soiled, 15 creased, 5 torn, and 14 folded. For the reviewed images, the human labels now replace the earlier values in the main annotation CSV; uncertain values remain masked during training. Authenticity labels, image paths, and split assignments were left unchanged.
- After the review, validation condition counts are: soiled 58 yes / 106 no / 50 uncertain; creased 173 / 15 / 26; torn 12 / 181 / 21; folded 152 / 34 / 28. The number of known torn-positive validation examples increased from 9 to 12, but remains small.

## Integrated multi-task model

- `src/banknote/models/multitask_multisignal_cnn.py` shares RGB, edge, and texture features across an authenticity head and four separate condition heads.
- `src/banknote/training/train_multitask_multisignal.py` trains the heads together, masks uncertain condition values, calculates capped positive weights from the training split, and selects checkpoints using combined validation loss. It does not load the test split for training or selection.
- Run the new workflow from the repository root with `python -m src.banknote.training.train_multitask_multisignal --cache-dir data/processed/streams_v1_224 --run-id condition-tags-seed42`.
- The initial 15-epoch seed-42 weight sweep used the same 998/214 train/validation split and the pre-review labels. Best-checkpoint authenticity accuracy was 85.98% at condition-loss weight 1.0, 88.32% at 0.5, and 89.25% at 0.25. Weight 1.0 favored the four condition heads, while 0.25 was closer to the 90.65% same-split single-head baseline. The equal task-loss balance (weight 1.0) was carried forward for repeat-seed assessment.
- The first three 15-epoch runs at weight 1.0 also used pre-review labels. Their results are retained below as the initial baseline; they are superseded by the reviewed-label runs in the next table.

  | Seed | Epoch | Authenticity accuracy | Soiled F1 | Creased F1 | Torn F1 | Folded F1 |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 42 | 13 | 0.860 | 0.522 | 0.789 | 0.727 | 0.727 |
  | 43 | 13 | 0.883 | 0.515 | 0.770 | 0.383 | 0.672 |
  | 44 | 14 | 0.907 | 0.515 | 0.735 | 0.243 | 0.699 |
  | Mean | n/a | 0.883 | 0.517 | 0.765 | 0.451 | 0.699 |

- After the labels changed, all three initial checkpoints were re-evaluated on validation. For seed 42, authenticity accuracy remained 0.860 while condition F1 changed from 0.522 to 0.535 for soiled, 0.789 to 0.793 for creased, 0.727 to 0.583 for torn, and 0.727 to 0.731 for folded. Each original run now has a `validation_reassessment.json` using the reviewed labels.
- Three new 15-epoch runs at weight 1.0 were trained on the reviewed annotations. Each used the same 998/214 train/validation split and selected its checkpoint by lowest combined validation loss. Best-checkpoint results at threshold 0.5 were:

  | Seed | Epoch | Validation loss | Authenticity accuracy | Soiled F1 | Creased F1 | Torn F1 | Folded F1 |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 42 | 7 | 1.1060 | 0.888 | 0.447 | 0.715 | 0.206 | 0.672 |
  | 43 | 13 | 1.0567 | 0.883 | 0.466 | 0.778 | 0.327 | 0.685 |
  | 44 | 14 | 1.0294 | 0.907 | 0.491 | 0.749 | 0.263 | 0.703 |
  | Mean | n/a | n/a | 0.893 | 0.468 | 0.747 | 0.265 | 0.687 |

- The selected checkpoint is seed 44, epoch 14, chosen by validation loss. These validation results use all 214 images, but only 39 have received human spot-review; remaining condition labels come from the earlier annotation pass. Treat condition metrics as preliminary, especially torn, which has 12 known positive validation examples.
- After validation selection, seed 44 was evaluated once on the 214-image held-out test split. Counterfeit detection accuracy was 0.907, precision 0.961, recall 0.860, F1 0.907, balanced accuracy 0.910, MCC 0.819, and ROC-AUC 0.963. Condition-head test F1 was 0.574 for soiled, 0.720 for creased, 0.060 for torn, and 0.659 for folded. Test condition labels have not received a human spot-review, and torn has only 2 known positive examples. Treat those condition metrics as exploratory. Do not use test results to choose another model or tune thresholds.
- The reviewed-label run folders are `runs/multitask/condition-human-review-seed42-20261127`, `runs/multitask/condition-human-review-seed43-20261127`, and `runs/multitask/condition-human-review-seed44-20261127`. The selected run contains `test_evaluation.json` and `test_confusion_matrix.png`.
- No test images were loaded for training or model selection. The held-out test evaluation is complete for the selected configuration.

## Still needs a project decision or new data

- Optional W&B tracking is not enabled. Local run folders already record metrics and configuration; W&B would send selected metadata to an external service and should only be enabled deliberately.
- A project-level provisional circulation rule is now documented in `annotations/prototype_fitness_rule.md`; it is a research proxy pending supervisor review, not an official BoU machine-sorting standard. The four visible condition outputs alone do not establish fitness.
- Initial low-confidence prelabels are in `annotations/prototype_fitness_prelabels.csv`: 760 counterfeit, 476 genuine-fit proxies, 90 genuine-unfit proxies, and 100 cases held for review. The 476 fit proxies only mean that no severe trigger was inferred from the four existing condition notes. The new damage flags are mostly unassessed. Review these labels before training or presenting them as ground truth.
- The condition-only checkpoint still has authenticity and four visible-condition outputs. The new concatenation prototype adds a genuine-only fit/unfit head and severity heads, but its weak-label results are not ready for operational use. Improve the labels and validate the final outcome on stronger human-reviewed or external data.

The seed-matched attention-versus-concatenation comparison is complete. Use validation results for selection; reserve test metrics for final reporting.

- The seed-42, 15-epoch hybrid comparison is complete on the same 998/214 split and with the same core settings. Concatenation achieved 93.46% validation accuracy and 0.1768 validation loss; attention achieved 91.59% and 0.1839. Concatenation is the current fusion choice based on validation. These are single-run results, so the difference is preliminary rather than evidence of a general attention advantage or disadvantage. The comparison used validation records only; existing test reports were not used for the choice.
- A supervisor-ready metrics deck is available at `outputs/01a21999-bdc7-73f3-bf42-5bf8235c948e/ugx_fitness_counterfeit_evaluation_2026-11-27_final.pptx` with a matching PDF. It summarizes the benchmark, signal ablation, label review, multi-task seed results, final counterfeit test metrics, and remaining limitations.

## Three-outcome prototype experiment

- `src/banknote/models/fitness_multitask_multisignal_cnn.py` adds a genuine-only fitness head (`fit`/`unfit`) alongside authenticity, the four condition-presence heads, and four severity heads. Its shared RGB/edge/texture encoder composes the three final outcomes and enforces counterfeit priority. It uses concatenation; the attention-based integrated version remains future work.
- `src/banknote/training/train_fitness_multitask.py` trains on low-confidence prelabels and excludes `review_required` rows. Run `python -m src.banknote.training.train_fitness_multitask --dataset-dir <labelled-dataset> --cache-dir <stream-cache> --run-id <run-name>`. Selection uses weighted validation outcome loss; the test split is opened only after checkpoint selection.
- Seed 42, 15 epochs selected epoch 13 from 935 training and 192 validation rows. Validation outcome accuracy was 0.797, macro-F1 0.688, and balanced accuracy 0.720. The one held-out proxy evaluation used 199 rows (114 counterfeit, 70 fit proxies, 15 unfit proxies): accuracy 0.804, macro-F1 0.692, balanced accuracy 0.724. Per-class test F1 was 0.906 for counterfeit, 0.732 for genuine-fit, and 0.439 for genuine-unfit. Unfit precision was 0.346 and recall 0.600.
- Test severity macro-F1 was 0.262 soiled, 0.286 creased, 0.497 torn, and 0.163 folded. The torn score is based on only two positive test labels; the severity outputs are not reliable enough to provide user-facing reasons such as “heavily creased.” All outcome and severity metrics measure agreement with low-confidence rule-based labels, not independent expert judgments or BoU machine decisions.
- The policy-corrected run artifacts are in `runs/fitness/prototype-fitness-policy-seed42-20261127/`, including configuration, validation history, selected checkpoint, test metrics, and test predictions. The checkpoint remains local because model weights are excluded from Git. An earlier independent three-class-head experiment is superseded because it did not hard-enforce counterfeit priority.

## Intended decision and Streamlit delivery

- The target outcome is one of: `counterfeit`, `genuine and fit for circulation`, or `genuine and unfit for circulation`. Counterfeit takes priority even when the note also shows damage. For genuine notes, show condition reasons such as soiled, creased, torn, or folded.
- Keep the model's internal outputs separate: authenticity, fitness status or grade, and condition reasons. A final decision layer should return `counterfeit` whenever the authenticity result is counterfeit; otherwise it should combine the genuine result with the fitness output.
- The current condition annotations record presence (`yes`, `no`, `uncertain`), not severity. A reason such as `heavily creased` needs an agreed severity scale and reviewed severity labels. Do not infer severity from the current binary heads.
- The current multi-task model concatenates RGB, edge, and texture features. The separate attention model is a binary authenticity classifier. The target attention-based multi-task model, with fitness and condition/severity outputs, remains to be designed, trained, and compared against concatenation on validation data.
- Attention weights can report how the model weights RGB, edges, and texture. They do not by themselves explain why a note is unfit. The condition/severity outputs and the approved fitness rule should provide the displayed reasons.
- `app.py` now provides a local Streamlit workspace with overview metrics, image assessment, editable prelabel review, evaluation charts, and the project rule/roadmap. Run it from the project root with `python -m streamlit run app.py`. The app can use the local selected checkpoint and optionally a dataset directory supplied in its sidebar. Label edits stay in the session until downloaded; the source annotation file is not overwritten.
- The assessment page shows the composed three-way outcome and keeps the authenticity, fitness, condition, and severity outputs visible as separate signals. Treat this as a guarded research preview: all current targets are low-confidence rule-based prelabels, and the test metrics are agreement with those prelabels, not expert judgments or BoU machine decisions. The condition/severity signals are review prompts, not confirmed reasons.
- Next sequence: improve outcome and severity labels through additional human image review; repeat model selection and held-out evaluation with those corrected labels; then assess the attention-based integrated model against concatenation on validation data. Keep the app output labelled as a prototype until checked against a stronger human-reviewed or external test set. Compare fresh and current notes only when denomination, side, and design/print variant match; use the fresh note as a visual reference, not a raw pixel-distance fitness test.
