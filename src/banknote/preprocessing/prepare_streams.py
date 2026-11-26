
import cv2
import numpy as np

from .transforms import (
    load_banknote_image,
    extract_rgb,
    extract_edges,
    extract_texture
)


def resize_preserving_aspect(image, target_size=224):
    """
    Resize without stretching the banknote.
    Pad the remaining space to create a square image.
    """

    height, width = image.shape[:2]

    scale = target_size / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Grey padding, consistent with our data loader
    canvas = np.full(
        (target_size, target_size, 3),
        127,
        dtype=np.uint8
    )

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def prepare_banknote_streams(image_path, target_size=224):
    """
    Generate three aligned representations
    from one banknote image.
    """

    image = load_banknote_image(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Standardise original image
    image = resize_preserving_aspect(
        image,
        target_size
    )

    # Generate representations
    rgb = extract_rgb(image)

    edges = extract_edges(image)

    texture = extract_texture(image)

    # Validate spatial alignment
    assert rgb.shape[:2] == edges.shape
    assert rgb.shape[:2] == texture.shape

    return {
        "original": image,
        "rgb": rgb,
        "edges": edges,
        "texture": texture
    }
