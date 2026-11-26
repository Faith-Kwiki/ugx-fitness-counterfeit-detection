
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib
import csv

# Dataset location
DATASET = Path(
    "data/raw/Ugandan banknote image dataset for genuine and cou/labelled_ugandan_banknote_dataset_clean"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(folder):
    """Find all supported image files inside a folder."""
    if not folder.exists():
        return []

    return [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def identify_class(path):
    """Identify authenticity from the folder name."""
    for part in reversed(path.parts):
        name = part.lower()

        if name in {"genuine", "counterfeit"}:
            return name

    return "unknown"


def image_hash(path):
    """Create a hash to identify exact duplicate files."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_folder(name, folder):
    images = find_images(folder)

    classes = Counter(
        identify_class(path) for path in images
    )

    print(f"\n{name}")
    print("-" * 40)
    print("Total images:", len(images))
    print("Genuine:", classes["genuine"])
    print("Counterfeit:", classes["counterfeit"])
    print("Unknown labels:", classes["unknown"])

    return images
def inspect_split_distribution(split_images):

    print("\nSPLIT DISTRIBUTION")
    print("-" * 40)

    for split, paths in split_images.items():

        genuine = sum(
            1 for path in paths
            if identify_class(path) == "genuine"
        )

        counterfeit = sum(
            1 for path in paths
            if identify_class(path) == "counterfeit"
        )

        print(f"\n{split}")
        print("Genuine:", genuine)
        print("Counterfeit:", counterfeit)
        print("Total:", len(paths))

def main():

    if not DATASET.exists():
        print("DATASET NOT FOUND!")
        print("Check the DATASET path in this script.")
        return

    # Inspect the complete image collection
    all_images = inspect_folder(
        "ALL IMAGES",
        DATASET / "all_images"
    )

    # Inspect the existing data splits
    split_names = [
        "train_set",
        "validation_set",
        "test_set"
    ]

    split_images = {}

    for split in split_names:
        split_path = DATASET / "splits" / split

        split_images[split] = inspect_folder(
            split.upper(),
            split_path
        )
        inspect_split_distribution(split_images)  
    # Check image dimensions and exact duplicates
    print("\nIMAGE DETAILS")
    print("-" * 40)

    dimensions = Counter()
    hashes = {}

    for path in all_images:
        try:
            with Image.open(path) as image:
                dimensions[image.size] += 1

            digest = image_hash(path)
            hashes.setdefault(digest, []).append(str(path))

        except Exception as error:
            print("Could not inspect:", path, error)

    print("Image dimensions:")
    for dimension, count in dimensions.most_common(10):
        print(dimension, ":", count)

    duplicates = [
        paths for paths in hashes.values()
        if len(paths) > 1
    ]

    print("\nExact duplicate groups:", len(duplicates))

    # Check whether exact duplicates occur across splits
    split_hashes = {}

    for split, paths in split_images.items():
        split_hashes[split] = {
            image_hash(path): str(path)
            for path in paths
        }

    print("\nEXACT DUPLICATE OVERLAP BETWEEN SPLITS")
    print("-" * 40)

    split_list = list(split_hashes)

    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):

            first = split_list[i]
            second = split_list[j]

            overlap = (
                set(split_hashes[first])
                & set(split_hashes[second])
            )

            print(
                first, "vs", second,
                ":", len(overlap), "matching images"
            )

    # Save an image manifest
    manifest_path = Path("reports/dataset_manifest.csv")
    manifest_path.parent.mkdir(exist_ok=True)

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_path",
            "authenticity_label",
            "width",
            "height"
        ])

        for path in all_images:
            try:
                with Image.open(path) as image:
                    width, height = image.size

                writer.writerow([
                    str(path),
                    identify_class(path),
                    width,
                    height
                ])

            except Exception:
                continue

    print("\nManifest saved to:", manifest_path)
    print("\nDataset inspection complete!")


if __name__ == "__main__":
    main()
