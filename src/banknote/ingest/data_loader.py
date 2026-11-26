
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps


# --------------------------------
# 1. DATASET CONFIGURATION
# --------------------------------

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

IMAGE_SIZE = 224
BATCH_SIZE = 8

# Counterfeit = 0, Genuine = 1
CLASS_LABELS = {
    "counterfeit": 0,
    "genuine": 1
}


# --------------------------------
# 2. BANKNOTE DATASET CLASS
# --------------------------------

class BanknoteDataset(Dataset):

    def __init__(self, split):

        self.image_paths = []
        self.labels = []

        split_folder = DATASET / "splits" / split

        if not split_folder.exists():
            raise FileNotFoundError(split_folder)

        for class_name, label in CLASS_LABELS.items():

            class_folder = split_folder / class_name

            for image_path in class_folder.glob("*"):

                if image_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }:
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        if not self.image_paths:
            raise ValueError(
                f"No images found in {split}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        # Read image and convert to RGB
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            # Resize proportionally and pad to square
            image = ImageOps.pad(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BILINEAR,
                color=(127, 127, 127)
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            ).copy()

        # Convert pixel values from 0-255 to 0-1
        image_array /= 255.0

        # Change shape from HWC to CHW
        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image_array)

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return image_tensor, label_tensor


# --------------------------------
# 3. CREATE DATA LOADERS
# --------------------------------

def create_data_loaders():

    train_dataset = BanknoteDataset("train_set")
    validation_dataset = BanknoteDataset("validation_set")
    test_dataset = BanknoteDataset("test_set")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader, test_loader
