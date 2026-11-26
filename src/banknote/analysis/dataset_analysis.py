
from pathlib import Path
from collections import Counter
import re

RAW = Path("data/raw")

matches = list(
    RAW.glob(
        "Ugandan banknote*/labelled_ugandan_banknote_dataset_clean"
    )
)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one dataset folder, found {len(matches)}"
    )

DATASET = matches[0]

SPLITS = ["train_set", "validation_set", "test_set"]

DENOMINATIONS = [
    "1000",
    "2000",
    "5000",
    "10000",
    "20000",
    "50000",
]

def extract_denomination(filename):
    match = re.search(
        r"UGX_(?:genuine|counterfeit)_(\d+)",
        filename,
        re.IGNORECASE
    )

    if not match:
        return "Unknown"

    # We will inspect the filename first.
    # Do not assume the sequence number is the denomination.
    return "Filename sequence: " + match.group(1)

for split in SPLITS:
    print(f"\n{split.upper()}")
    print("-" * 40)

    folder = DATASET / "splits" / split

    for class_name in ["genuine", "counterfeit"]:
        class_folder = folder / class_name

        images = list(class_folder.glob("*.jpg"))

        print(f"\n{class_name}: {len(images)} images")

        sequence_numbers = Counter(
            extract_denomination(image.name)
            for image in images
        )

        for sequence, count in sequence_numbers.most_common():
            print(f"  {sequence}: {count}")
