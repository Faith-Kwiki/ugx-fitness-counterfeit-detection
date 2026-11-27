"""Build auditable prototype fitness prelabels from reviewed condition notes.

This creates weak labels for prototype exploration; it is not an independent
visual inspection and must not be presented as Bank of Uganda certification.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CONDITIONS = ("soiled", "creased", "torn", "folded")
YES = {"yes"}
NO = {"no"}
UNCERTAIN = {"uncertain", ""}

SEVERE_PATTERNS = {
    "soiled": re.compile(r"\b(heavy|heavily|severe|extensive)\s+(?:surface\s+)?(?:stain(?:ing)?|soiling|discoloration|wear|dirt|grease|marks?)\b", re.I),
    "creased": re.compile(r"\b(heavy|heavily|severe|deep)\s+(?:surface\s+)?(?:creasing|creases?|fold(?:ing|s)?|crumpling|wrinkling)\b", re.I),
    "torn": re.compile(r"\b(severe|heavy|heavily|large|deep|ragged|extensive)\s+(?:visible\s+)?(?:tear|torn|split|hole|edge|missing)", re.I),
    "folded": re.compile(r"\b(heavy|heavily|severe|deep)\s+(?:surface\s+)?(?:fold(?:ing|s)?|crumpling|wrinkling)\b", re.I),
}
MODERATE_PATTERNS = {
    "soiled": re.compile(r"\b(pronounced|widespread|multiple|heavy|severe|extensive)\b.{0,12}\b(stain|soiling|discolor|wear|dirt|grease|mark)", re.I),
    "creased": re.compile(r"\b(pronounced|multiple|intersecting|broad|strong|crumpl|wrinkl)\b.{0,18}\b(crease|fold|crumpl|wrinkl)", re.I),
    "torn": re.compile(r"\b(tear|torn|split|hole|ragged|edge damage|missing)", re.I),
    "folded": re.compile(r"\b(pronounced|multiple|intersecting|broad|strong|crumpl|wrinkl)\b.{0,18}\b(fold|crumpl|wrinkl)", re.I),
}
UNCLEAR_IMAGE = re.compile(r"blur|glare|soft.focus|cropp|obscur|unclear|not clear|limited detail|edge condition is unclear", re.I)


def severity_for(tag: str, presence: str, notes: str) -> str:
    value = presence.strip().lower()
    if value in NO:
        return "none"
    if value in UNCERTAIN:
        return "uncertain"
    if value not in YES:
        return "uncertain"
    if SEVERE_PATTERNS[tag].search(notes):
        return "severe"
    if MODERATE_PATTERNS[tag].search(notes):
        return "moderate"
    return "mild"


def explicit_flag(notes: str, pattern: str) -> str:
    return "yes" if re.search(pattern, notes, re.I) else "uncertain"


def prelabel(row: dict[str, str]) -> dict[str, str]:
    notes = (row.get("notes") or "").strip()
    old_auth = (row.get("authenticity_label") or "").strip().lower()
    result = dict(row)
    result["existing_authenticity_label"] = old_auth
    result["reviewed_authenticity"] = old_auth if old_auth in {"genuine", "counterfeit"} else "uncertain"
    result["photo_quality"] = "limited" if UNCLEAR_IMAGE.search(notes) else "condition_visible_only"

    for tag in CONDITIONS:
        presence = (row.get(tag) or "").strip().lower()
        result[f"{tag}_presence"] = presence if presence in {"yes", "no", "uncertain"} else "uncertain"
        result[f"{tag}_severity"] = severity_for(tag, presence, notes)

    result["adhesive_or_fasteners"] = explicit_flag(notes, r"tape|glue|adhesive|staple|pin(?:ned)?")
    result["moisture_damage"] = explicit_flag(notes, r"moist|damp|wet|water damage|water stain")
    result["defacement"] = explicit_flag(notes, r"writing|written|draw(?:ing)?|political message|marker|ink mark")
    result["important_details_obscured"] = explicit_flag(notes, r"security feature.{0,20}(obscur|cover)|denomination.{0,20}obscur|serial.{0,20}obscur|details.{0,20}obscur")

    # The four existing condition tags support only a visible-condition proxy.
    # Redemption tests need scale and complete note features, absent from the
    # annotation records, so leave those assessments explicitly unavailable.
    for key in (
        "redemption_size_at_least_two_thirds",
        "redemption_signature_complete",
        "redemption_denomination_complete",
        "redemption_serial_complete",
    ):
        result[key] = "not_assessable"

    severities = [result[f"{tag}_severity"] for tag in CONDITIONS]
    any_trigger = any(
        result[key] == "yes"
        for key in ("adhesive_or_fasteners", "moisture_damage", "defacement", "important_details_obscured")
    )
    if old_auth == "counterfeit":
        result["fitness_outcome"] = "counterfeit"
    elif old_auth != "genuine":
        result["fitness_outcome"] = "review_required"
    elif any_trigger or "severe" in severities:
        result["fitness_outcome"] = "genuine_unfit"
    elif "uncertain" in severities:
        result["fitness_outcome"] = "review_required"
    else:
        result["fitness_outcome"] = "genuine_fit"

    result["reviewer"] = "prototype_note_prelabeler"
    result["confidence"] = "low"
    result["notes"] = (
        "PROVISIONAL PRELABEL: copied from existing authenticity/condition tags and their note text; "
        "severity is inferred from wording, not freshly verified image by image. Extra damage flags are "
        "uncertain unless explicitly mentioned. A genuine_fit outcome means no severe trigger was "
        "identified among the four existing condition tags; unrecorded triggers remain unverified. "
        "Review before training or reporting. "
        + ("Source note: " + notes if notes else "Source note is blank.")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("annotations/condition_labels_template.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotations/prototype_fitness_prelabels.csv"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional root of the labelled dataset folder containing splits/; adds direct local image links.",
    )
    args = parser.parse_args()

    with args.source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        source_rows = list(reader)
        output_fields = list(reader.fieldnames or [])
    required = {
        "image_id", "image_path", "split", "authenticity_label", "soiled", "creased", "torn", "folded", "notes"
    }
    missing = required.difference(output_fields)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {', '.join(sorted(missing))}")

    extra_fields = [
        "condition_annotator", "condition_annotation_confidence", "condition_annotation_reviewer",
        "existing_authenticity_label", "denomination", "issue_or_series", "note_side", "photo_quality",
        "reviewed_authenticity", "fitness_outcome", "soiled_presence", "soiled_severity", "creased_presence",
        "creased_severity", "torn_presence", "torn_severity", "folded_presence", "folded_severity",
        "adhesive_or_fasteners", "moisture_damage", "defacement", "important_details_obscured",
        "redemption_size_at_least_two_thirds", "redemption_signature_complete",
        "redemption_denomination_complete", "redemption_serial_complete", "image_link",
        "prelabel_reviewer", "prelabel_confidence",
    ]
    fields = [*extra_fields, "prelabel_notes"]
    output_fields = [
        field for field in output_fields if field not in {"annotator", "confidence", "reviewer", "notes"}
    ] + fields
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for source in source_rows:
            labeled = prelabel(source)
            labeled["condition_annotator"] = source.get("annotator", "")
            labeled["condition_annotation_confidence"] = source.get("confidence", "")
            labeled["condition_annotation_reviewer"] = source.get("reviewer", "")
            labeled["prelabel_reviewer"] = labeled.pop("reviewer")
            labeled["prelabel_confidence"] = labeled.pop("confidence")
            labeled["prelabel_notes"] = labeled.pop("notes")
            labeled["image_link"] = ""
            if args.dataset_root:
                source_path = (source.get("image_path") or "").replace("\\", "/")
                if "/splits/" not in source_path:
                    raise ValueError(f"Image path does not include a splits/ location: {source_path}")
                relative_image = Path("splits") / source_path.split("/splits/", 1)[1]
                image_file = (args.dataset_root / relative_image).resolve()
                if not image_file.is_file():
                    raise FileNotFoundError(f"Dataset image not found: {image_file}")
                labeled["image_path"] = str(image_file)
                labeled["image_link"] = image_file.as_uri()
            for key in ("denomination", "issue_or_series", "note_side"):
                labeled[key] = ""
            writer.writerow(labeled)

    print(f"Wrote {len(source_rows)} provisional prelabels to {args.output}")
    counts: dict[str, int] = {}
    for row in source_rows:
        outcome = prelabel(row)["fitness_outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    print("Outcome counts:")
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count}")


if __name__ == "__main__":
    main()
