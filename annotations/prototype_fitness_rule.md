# Prototype circulation fitness rule

**Status:** working research rule for this prototype; pending supervisor review. It is not an official Bank of Uganda sorter standard.

## Intended decision

The final display has three outcomes:

- `counterfeit`
- `genuine and fit for circulation`
- `genuine and unfit for circulation`

An image that cannot be assessed should be sent for review instead of forcing one of those labels. The authenticity decision takes priority: a counterfeit note remains counterfeit even when it also shows damage. Condition reasons can still be recorded separately.

## Provisional circulation rule

For a note assessed as genuine, mark it **unfit** when the visible evidence shows any of these:

- Glue, tape, staples, pins, or other attached material used to repair or decorate the note.
- A severe tear, hole, or missing piece that compromises the note's structure or obscures important printed or security details.
- Severe creasing, crumpling, or folding that visibly compromises the note's structure or makes important details hard to inspect.
- Severe dirt, grease, charcoal, sweat staining, or visible moisture damage.
- Added writing, drawing, or other defacement.

Mark it **fit under this prototype rule** when the image is clear enough to assess and none of those unfit triggers is present. Record mild or moderate wear as a condition reason without automatically calling the note unfit. If image quality or the evidence is insufficient, use `review_required`/`uncertain` rather than guessing.

This rule is a practical first pass based on the project criteria supplied by the researcher. It should be checked against a supervisor's judgment and revised after a reviewed sample. It cannot establish whether a physical note will pass a banknote sorting machine.

## Condition and severity labels

Keep the existing four condition tags (`soiled`, `creased`, `torn`, `folded`) as separate observations. For the next annotation pass, record presence (`yes`, `no`, `uncertain`) and severity (`none`, `mild`, `moderate`, `severe`, `uncertain`) for each. Use `uncertain` when the image does not provide enough evidence. Severity descriptions are provisional:

- **Mild:** visible but small/localized wear with the printed details and note structure intact.
- **Moderate:** clear wear across a larger area, while the note remains structurally intact and important details remain readable.
- **Severe:** substantial structural damage, obscured important details, or widespread staining/moisture damage.

Presence and severity describe the image; they do not by themselves prove authenticity or machine fitness. Add separate yes/no/uncertain flags for adhesive/fasteners, moisture damage, defacement, and important details obscured. Reasons shown to a user should come from these reviewed labels and the rule, not from attention weights.

## Keep redemption eligibility separate

BoU's public educational comic describes redemption conditions for damaged notes: at least two-thirds of the original size, one complete Governor or Bank Secretary signature, one full denomination figure, and one complete serial number. BoU also reserves the right to assess each note. These are redemption checks, not the prototype's circulation-fitness rule. Record each as `yes`, `no`, or `uncertain` only when the relevant area is visible and the image supports the judgment. Otherwise use `not_assessable`.

## Comparing a fresh note with a current note

A fresh BoU note can help a reviewer locate the expected layout and features. For a useful comparison, match denomination, note side, and design/print variant, and capture both notes at similar scale and lighting. The current project photos vary in angle, illumination, background, and framing, so raw pixel differences against one fresh-note photo are not a reliable fitness test. Compare visible structure and features with human review; do not treat a single reference image as ground truth.

## Use in the project

The current trained model predicts authenticity and four condition-presence labels. It does not yet predict condition severity or an approved fitness outcome. Do not describe the current checkpoint as making the three-way decision. The next model work needs reviewed severity and fitness labels, a validation-only rule/model choice, and a separate untouched evaluation set for the new outcome.
