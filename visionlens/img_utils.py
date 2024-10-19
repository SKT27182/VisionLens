import IPython
import einops
import numpy as np
from typing import List, Optional, Union
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision.transforms.functional import to_pil_image

from visionlens.utils import T, create_logger

logger = create_logger(__name__)


def tensor_to_img_array(tensor: T) -> Image:
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
    """Create an HTML table cell with an image and title above it."""
    logger.debug(f"Creating table cell for image of shape {image.shape}.")
    style = f"margin: {margin}px; text-align: center;"
    cell_html = f'<td style="{style}">'
    if title:
        cell_html += f"<h4 style='margin-bottom: 5px;'>{title}</h4>"
    cell_html += _single_image_html(image, width)
    cell_html += "</td>"
    return cell_html


def create_image_table(
    images: T,
    labels: Optional[Union[List[str], str]] = None,
    width: Optional[int] = None,
    n_rows: Union[int, None] = None,
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

    labels = labels[0] if (isinstance(labels, list) and (len(labels) == 1)) else labels

    for i in range(n_rows):
        table_html += "<tr>"
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(images):
                logger.debug(f"Creating table cell for image {idx}.")
                if isinstance(labels, list):
                    title = labels[idx]
                elif isinstance(labels, str):
                    title = f"{labels} {idx + 1}"
                else:
                    title = ""
                table_html += image_to_html_table_cell(
                    images[idx], width, title, margin
                )
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


def display_images_in_table(
    images: T,
    labels: Optional[Union[List[str], str]] = None,
    width: Optional[int] = None,
    n_rows: int = 1,
    margin: Optional[int] = 5,
) -> None:
    """Display a list of images in a table format in a Jupyter notebook."""

    logger.info(f"Displaying {images.shape[0]} images in a table.")

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
    quality: int = 100,
) -> None:
    """Save an image to a file."""

    img = tensor_to_img_array(image)

    logger.info(f"Saving image to {file_path}.")

    img.save(file_path, format=fmt, quality=quality)


def _save_images(
    images: T,
    dir_path: str,
    file_names: Union[List[str], str],
    fmt: str = "PNG",
    quality: int = 100,
) -> None:
    """Save a list of images to files."""

    for i, img in enumerate(images):
        file_name = (
            file_names[i] if isinstance(file_names, list) else f"{file_names}_{i}"
        )
        file_path = f"{dir_path}/{file_name}.{fmt.lower()}"
        save_image(img, file_path, fmt, quality)


# Assuming images is a 4D numpy array with shape (num_images, channels, height, width)
def save_images_as_table(images, n_rows, output_path, padding=10, labels=None):
    num_images, _, height, width = images.shape

    # if n_rows is not provided, will calculate so that the table is square
    n_rows = n_rows if n_rows else int(np.ceil(np.sqrt(len(images))))
    n_cols = len(images) // n_rows
    if n_rows * n_cols < len(images):
        n_rows += 1

    # Create a blank canvas with padding
    table_height = n_rows * (height + padding + 20) - padding
    table_width = n_cols * (width + padding) - padding

    table_image = Image.new("RGB", (table_width, table_height))

    draw = ImageDraw.Draw(table_image)
    font = ImageFont.load_default()

    y_offset = 0

    for idx in range(num_images):
        img = to_pil_image(images[idx])
        row = idx // n_cols
        col = idx % n_cols
        x = col * (width + padding)
        y = row * (height + padding + 20) + y_offset
        table_image.paste(img, (x, y))

        # Add title
        img_title = labels[idx] if labels and idx < len(labels) else f"Image_{idx + 1}"
        text_width, text_height = draw.textbbox((0, 0), img_title, font=font)[2:]
        text_x = x + (width - text_width) // 2
        text_y = y + height + 5
        draw.text((text_x, text_y), img_title, fill="white", font=font)

    table_image.save(output_path, format="PNG", quality=100)
