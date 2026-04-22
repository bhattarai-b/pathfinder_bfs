import numpy as np
from PIL import Image, ImageDraw


def load_image(path: str, threshold: int = 128) -> np.ndarray:
    """Load a PNG image and binarize it.

    Args:
        path: Path to the image file.
        threshold: Pixel values below this become 0 (black/traversable),
                   at or above become 255 (white/wall).

    Returns:
        2D numpy array with values 0 (black) or 255 (white).
    """
    img = Image.open(path).convert("L")  # grayscale
    arr = np.array(img)
    binary = np.where(arr < threshold, 0, 255).astype(np.uint8)
    return binary


def save_annotated(
    image_path: str,
    path: list[tuple[int, int]],
    output_path: str,
    start: tuple[int, int] | None = None,
    end: tuple[int, int] | None = None,
    path_color: tuple[int, int, int] = (255, 0, 0),
    start_color: tuple[int, int, int] = (0, 255, 0),
    end_color: tuple[int, int, int] = (0, 0, 255),
    endpoint_radius: int | None = None,
):
    """Draw the path and endpoints on the original image and save it.

    Args:
        image_path: Path to the source PNG (loaded and converted to RGB).
        path: List of (row, col) tuples forming the path.
        output_path: Where to save the annotated PNG.
        start: Optional start coordinate — drawn as a green dot.
        end: Optional end coordinate — drawn as a blue dot.
        path_color: RGB color for path pixels (default red).
        start_color: RGB color for start marker (default green).
        end_color: RGB color for end marker (default blue).
        endpoint_radius: Radius of the start/end markers in pixels.
                         None = auto-scale based on image size.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = img.load()

    # auto-scale endpoint radius: ~1% of the smaller dimension, min 0
    if endpoint_radius is None:
        endpoint_radius = max(0, min(img.width, img.height) // 100)

    # draw path
    for r, c in path:
        pixels[c, r] = path_color  # PIL uses (x, y) = (col, row)

    # draw endpoint markers
    if start or end:
        draw = ImageDraw.Draw(img)
        rad = endpoint_radius
        if start:
            sr, sc = start
            if rad == 0:
                pixels[sc, sr] = start_color
            else:
                draw.ellipse([sc - rad, sr - rad, sc + rad, sr + rad], fill=start_color)
        if end:
            er, ec = end
            if rad == 0:
                pixels[ec, er] = end_color
            else:
                draw.ellipse([ec - rad, er - rad, ec + rad, er + rad], fill=end_color)

    img.save(output_path)
    return output_path
