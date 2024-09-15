import IPython
import einops
import numpy as np
from typing import List, Optional
import base64
from io import BytesIO

import torch
from torchvision.transforms.functional import to_pil_image

from visionlens.utils import T, create_logger

logger = create_logger(__name__)


def tensor_to_img_array(tensor: T) -> T:
    tensor = tensor.cpu().detach()

    if (len(tensor.shape) == 4) & (tensor.shape[0] == 1):  # (1 C H W)
        logger.warning(
            f"Converting {tensor.shape} tensor to 3D tensor by removing first dimension."
        )
        tensor = einops.rearrange(tensor, "1 c h w -> c h w")


    img = to_pil_image(tensor, mode="RGB")

    return img


def convert_arr_to_base64_str(arr: T, format="PNG", quality: int = 80) -> str:
    """Convert a NumPy array image to a base64 string."""

    img = tensor_to_img_array(arr)

    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return "data:image/" + format.upper() + ";base64," + img_str


def _single_image_html(
    image: T,
    width: Optional[int] = None,
    fmt: str = "png",
    quality: int = 85,
) -> str:
    """Convert a single image to an HTML image element."""
    img_url = convert_arr_to_base64_str(image, fmt, quality)
    style = f"image-rendering: pixelated; width: {width}px;" if width else ""
    return f'<img src="{img_url}" style="{style}">'


def image_to_html_table_cell(
    image: T,
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
    images: List[T],
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
    images: List[T],
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


def display_image(
    image: T,
    title: Optional[str] = "",
    width: Optional[int] = None,
    fmt: str = "png",
    quality: int = 85,
) -> None:
    """Display a single image in a Jupyter notebook."""

    html_str = _single_image_html(image, width, fmt, quality)
    if title:
        html_str += f"<h4>{title}</h4>"

    IPython.display.display(IPython.display.HTML(html_str))


def save_image(
    image: T,
    file_path: str,
    fmt: str = "PNG",
    quality: int = 85,
) -> None:
    """Save an image to a file."""

    img = tensor_to_img_array(image)

    img = to_pil_image(img.to(torch.uint8), mode="RGB")
    logger.info(f"Saving image to {file_path}.")

    img.save(file_path, format=fmt, quality=quality)
