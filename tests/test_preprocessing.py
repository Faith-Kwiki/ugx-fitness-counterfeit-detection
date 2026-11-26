import matplotlib.pyplot as plt

from preprocessing import (
    load_banknote_image,
    extract_rgb,
    extract_edges,
    extract_texture
)

# Replace this with the path to an image you have permission to use.
image_path = "data/raw/test_banknote.jpg"

image = load_banknote_image(image_path)

rgb = extract_rgb(image)
edges = extract_edges(image)
texture = extract_texture(image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb)
plt.title("RGB Representation")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Structural Representation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(texture, cmap="gray")
plt.title("Texture Representation - LBP")
plt.axis("off")

plt.tight_layout()
plt.show()