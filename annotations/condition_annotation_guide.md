# Image condition annotation guide

This first annotation pass records visible banknote condition tags. It does not assign a joint fitness/authenticity class.

## Start the image labeler

From the project folder, run:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\annotate_conditions.py
~~~

Then open http://127.0.0.1:8765 in your browser. The browser page shows one image at a time. Keep the PowerShell window open while annotating; press Ctrl+C there when finished.

## For each image

Choose yes, no, or uncertain for each tag. Tags are independent: an image can have several yes values. Use uncertain when the image is too blurry, cropped, shadowed, or otherwise unclear. The page saves when you click Save or move to another image. An image is marked complete after all four tags have values. Annotator name, confidence, and notes are optional. The page uses the existing split and authenticity label only as context.

## Working definitions for this project

- **Soiled:** visible dirt, grime, or staining on the note. Do not count normal printed color or lighting/shadow as soil.
- **Creased:** a visible line or compressed ridge left by bending or pressure, while the note is not currently folded over itself.
- **Torn:** a visible split, hole, or missing piece of note material. Mark uncertain if resolution prevents confirmation.
- **Folded:** the note is visibly bent over itself or has a clear fold shape in the image. A flat note that only shows an old crease is creased, not automatically folded.

These are image-level visual tags for annotation consistency, not an official Bank of Uganda fitness standard. The template contains existing authenticity labels and split names as context; those labels must not be changed. The model should not learn from unreviewed blank tags. After condition labels are reviewed, they can supervise separate condition outputs while the authenticity output is retained. A combined class can be considered later, once the annotations and labeling rule have been reviewed.

## Validation spot-review packet

`condition_spot_review_blind.csv` contains a 39-image validation-only sample. It includes all 9 validation images currently tagged torn, 9 torn-negative examples balanced across authenticity labels, and additional positive and negative examples for soiled, creased, and folded. Some images represent more than one tag group.

Review the images using the definitions above and enter `yes`, `no`, or `uncertain` in the four `review_` columns. Record reviewer name, confidence (`low`, `medium`, or `high`), and any short explanation. Do not open `condition_spot_review_key.csv` until the blind review is complete. Then compare by `review_id`, note disagreements, and update the main annotation file only after resolving them. This packet does not include test images, and completing it does not by itself establish an official fitness rating.

## Prototype fitness review

The completed condition annotation file remains unchanged. For the next pass, use `prototype_fitness_review_template.csv` with `prototype_fitness_rule.md`. The new template carries image identifiers, paths, splits, and the existing authenticity label as context; all new review fields are blank. Confirm or correct authenticity in `reviewed_authenticity`, then enter condition presence and severity, the overall prototype fitness outcome, and the separate redemption checks. Use `uncertain` when the image does not support a judgment and `not_assessable` for redemption checks whose required features are not visible. Do not use test-split labels to change the rule or select a model; lock the rule using training/validation review and reserve test data for evaluation.
