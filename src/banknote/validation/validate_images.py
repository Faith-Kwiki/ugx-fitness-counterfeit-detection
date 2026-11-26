
from pathlib import Path
from PIL import Image
from collections import Counter

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

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

valid_images = 0
corrupted_images = []
dimension_counts = Counter()
aspect_ratios = Counter()

for image_path in (DATASET / "splits").rglob("*"):

    if image_path.suffix.lower() not in image_extensions:
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            width, height = img.size

            valid_images += 1
            dimension_counts[(width, height)] += 1

            ratio = round(width / height, 2)
            aspect_ratios[ratio] += 1

    except Exception as error:
        corrupted_images.append(
            (image_path.name, str(error))
        )

print("\nIMAGE VALIDATION RESULTS")
print("-" * 40)

print("Valid images:", valid_images)
print("Corrupted images:", len(corrupted_images))

print("\nMost common dimensions:")
for dimensions, count in dimension_counts.most_common(10):
    print(dimensions, ":", count)

print("\nAspect ratios:")
for ratio, count in aspect_ratios.most_common():
    print(ratio, ":", count)

if corrupted_images:
    print("\nCORRUPTED FILES")
    for filename, error in corrupted_images:
        print(filename, error)
