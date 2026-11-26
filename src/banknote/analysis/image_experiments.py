
import cv2
import matplotlib.pyplot as plt

from preprocessing import (
    load_banknote_image,
    extract_edges,
    extract_texture
)

image_path = "data/raw/test_banknote.jpg"

image = load_banknote_image(image_path)

# The longest side will be resized to each target size.
sizes = [128, 224, 448]

plt.figure(figsize=(12, 10))

for i, size in enumerate(sizes):

    height, width = image.shape[:2]

    scale = size / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    edges = extract_edges(resized)
    texture = extract_texture(resized)

    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(edges, cmap="gray")
    plt.title(f"Edges - {size}")
    plt.axis("off")

    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(texture, cmap="gray")
    plt.title(f"LBP - {size}")
    plt.axis("off")

plt.tight_layout()
plt.show()
