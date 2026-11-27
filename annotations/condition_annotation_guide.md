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
