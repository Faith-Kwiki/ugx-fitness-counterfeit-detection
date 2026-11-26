import cv2
import numpy as np


def load_banknote_image(image_path):
    """Load a banknote image and convert it to RGB."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def extract_rgb(image):
    """Return the original colour representation."""
    return image.copy()


def extract_edges(image):
    """Extract structural information using Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        gray,
        threshold1=100,
        threshold2=200
    )

    return edges


def extract_texture(image):
    """Generate an initial Local Binary Pattern representation."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = np.zeros_like(gray, dtype=np.uint8)

    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):

            center = gray[y, x]

            neighbors = [
                gray[y-1, x-1],
                gray[y-1, x],
                gray[y-1, x+1],
                gray[y, x+1],
                gray[y+1, x+1],
                gray[y+1, x],
                gray[y+1, x-1],
                gray[y, x-1]
            ]

            value = 0

            for i, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    value |= (1 << i)

            lbp[y, x] = value

    return lbp