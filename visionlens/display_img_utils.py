import IPython
import einops
import numpy as np
from PIL import Image
from typing import List, Optional
import base64
from io import BytesIO

from visionlens.utils import A


def convert_arr_to_base64_str(arr: A, format="PNG", quality: int = 80) -> str:
    """Convert a NumPy array image to a base64 string."""

    # if tensor is 4D (1, C, H, W), convert to 3D (C, H, W)

    if len(arr.shape) == 4:
        arr = einops.rearrange(arr, "1 c h w -> h w c")

    # if pytorch tensor, convert to numpy array
    if "torch" in str(type(arr)):
        arr = arr.cpu().detach().numpy()

    # if the image is in the range [0, 1], convert to [0, 255]
    if (arr.max() <= 1) & (arr.min() >= 0):
        arr = arr * 255

    # if the image is in the range [-1, 1], convert to [0, 255]
    if (arr.max() <= 1) & (arr.min() >= -1):
        arr = (arr + 1) * 127.5

    img = Image.fromarray(arr.astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return "data:image/" + format.upper() + ";base64," + img_str


def _single_image_html(
    image: A,
    width: Optional[int] = None,
    fmt: str = "png",
    quality: int = 85,
) -> str:
    """Convert a single image to an HTML image element."""
    img_url = convert_arr_to_base64_str(image, fmt, quality)
    style = f"image-rendering: pixelated; width: {width}px;" if width else ""
    return f'<img src="{img_url}" style="{style}">'


def image_to_html_table_cell(
    image: np.ndarray,
    width: Optional[int] = None,
    title: Optional[str] = "",
    margin: Optional[int] = 5,
) -> str:
    """Create an HTML table cell with an image."""
    style = f"margin: {margin}px;"
    cell_html = f'<td style="{style}">'
    cell_html += _single_image_html(image, width)
    if title:
        cell_html += f"<h4>{title}</h4>"
    cell_html += "</td>"
    return cell_html


def create_image_table(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    width: Optional[int] = None,
    n_rows: int = None,
    margin: Optional[int] = 5,
) -> str:
    """Create an HTML table of images."""

    # if n_rows is not provided, will calculate so that the table is square
    n_rows = n_rows if n_rows else int(np.ceil(np.sqrt(len(images))))
    n_cols = len(images) // n_rows
    if n_rows * n_cols < len(images):
        n_rows += 1

    table_html = """<style>
    td:hover {
        transition: transform 0.5s;
        transform: scale(1.1);
    }
    </style>
    <table style="border-collapse: collapse;">
    """

    for i in range(n_rows):
        table_html += "<tr>"
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(images):
                title = labels[idx] if labels else ""
                table_html += image_to_html_table_cell(
                    images[idx], width, title, margin
                )
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


def display_images_in_table(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    width: Optional[int] = None,
    n_rows: int = 1,
    margin: Optional[int] = 5,
) -> None:
    """Display a list of images in a table format in a Jupyter notebook."""

    html_str = '<div style="display: flex; flex-wrap: wrap;">'
    img_html_str = create_image_table(images, labels, width, n_rows, margin)

    html_str += f"""<div style="margin: 10px;">{img_html_str}</div>"""
    html_str += "</div>"

    IPython.display.display(IPython.display.HTML(html_str))
