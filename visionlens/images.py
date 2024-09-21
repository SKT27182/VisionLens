from typing import Callable, List
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize
import einops
import numpy as np


from visionlens.utils import T, device, create_logger

logger = create_logger(__name__)

color_correlation_svd_sqrt = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]


def _linear_decorrelate_color(tensor):
    t_permute = einops.rearrange(tensor, "b c h w -> b h w c")
    t_permute = torch.matmul(
        t_permute, torch.tensor(color_correlation_normalized.T).to(t_permute.device)
    )
    tensor = einops.rearrange(t_permute, "b h w c -> b c h w")
    logger.debug(
        f"Returning decorrelated {tensor.dtype} tensor of shape {tensor.shape}"
    )
    return tensor


def to_valid_rgb(image_f, decorrelate=False):
    def inner():
        image = image_f()
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)

    return inner


def pixel_image(shape, sd=None, device=device):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1, device=device):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, channels) + freqs.shape + (2,)
    )  # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        try:  # torch.__version__ >= "1.7.0"
            logger.debug("Using torch.fft.irfftn, as torch_version is >= 1.7.0")
            import torch.fft

            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        except:
            import torch

            logger.debug("Using torch.irfft, as torch_version is < 1.7.0")

            image = torch.irfft(
                scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w)
            )
        image = image[:batch, :channels, :h, :w]
        magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


def get_images(
    w,
    h=None,
    sd=None,
    batch=None,
    decorrelate=True,
    fft=True,
    channels=None,
    device=device,
):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output


###### transforms
def compose(transforms: List[Callable[[T], T]]) -> Callable[[T], T]:
    """
    Composes a list of transforms into a single transform function.

    Args:
        transforms (list): A list of transform functions.

    Returns:
        function: A composed transform function that applies each transform in the given order.

    Example:
        transforms = [transform1, transform2, transform3]
        composed_transform = compose(transforms)
        result = composed_transform(input_image)
    """

    def inner(x):

        if not transforms:
            return x
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def random_pad_image(pad_size: int, padding_modes: List[str], fill: float):
    """
    Randomly pads an image with specified pad size, padding modes, and fill value.
    Args:
        pad_size (int or tuple): The size of the padding. If it is a single integer, the padding will be symmetric on all sides. If it is a tuple of 4 integers, it represents the padding on the left, top, right, and bottom sides respectively.
        padding_modes (list): A list of padding modes to choose from randomly. The padding mode determines how the padding pixels are filled. Examples of padding modes are 'constant', '', 'reflect', and 'wrap'.
        fill (int or tuple): The value or tuple of values to fill the padding pixels with. If it is a single integer, all padding pixels will be filled with that value. If it is a tuple of 3 integers, it represents the RGB values to fill the padding pixels with.
    Returns:
        inner (function): A function that takes an image tensor as input and returns the padded image tensor.
    """

    def inner(image_t: T):
        padding_mode = np.random.choice(padding_modes)
        logger.debug(
            f"Padding image with {pad_size} pixels using {padding_mode} padding mode"
        )
        return F.pad(image_t, padding=[pad_size], padding_mode=padding_mode, fill=fill)

    return inner


def random_jitter_image(dx: List):
    """
    Applies random jitter to an image.
    Args:
        dx (int): The maximum displacement in the x-axis.
    Returns:
        inner (function): A function that applies random jitter to an image.
    Raises:
        AssertionError: If dx is less than or equal to 1.
    Example:
        >>> jitter = random_jitter_image(5)
        >>> image = Image.open('image.jpg')
        >>> jittered_image = jitter(image)
    """

    assert all(x >= 1 for x in dx), "All elements of dx must be greater than 1"

    def inner(image_t: T):
        dx_ = np.random.choice(dx)
        dy_ = np.random.choice(dx)
        logger.debug(
            f"Jittering image by {dx_} pixels in x-axis and {dy_} pixels in y-axis"
        )
        return F.affine(image_t, angle=0, translate=(dx_, dy_), scale=1, shear=0)

    return inner


def random_scale_image(scales: List[float]):
    """
    Randomly scales an image by a factor chosen from a list of scales.
    Args:
        scales (list): A list of scaling factors to choose from randomly.
    Returns:
        inner (function): A function that takes an image tensor as input and returns the scaled image tensor.
    Example:
        >>> scale = random_scale_image([0.9, 1.1])
        >>> image = Image.open('image.jpg')
        >>> scaled_image = scale(image)
    """

    def inner(image_t: T):
        scale_factor = np.random.choice(scales)
        logger.debug(f"Scaling image by {scale_factor}")
        return F.affine(image_t, angle=0, translate=[0, 0], scale=scale_factor, shear=0)

    return inner


def random_rotate_image(degrees: List[int]):
    """
    Randomly rotates an image by an angle chosen from a list of degrees.
    Args:
        degrees (list): A list of rotation angles to choose from randomly.
    Returns:
        inner (function): A function that takes an image tensor as input and returns the rotated image tensor.
    Example:
        >>> rotate = random_rotate_image([-10, 0, 10])
        >>> image = Image.open('image.jpg')
        >>> rotated_image = rotate(image)
    """

    def inner(image_t: T):
        angle = np.random.choice(degrees).astype(float)
        logger.debug(f"Rotating image by {angle} degrees")
        return F.affine(image_t, angle=angle, translate=[0, 0], scale=1, shear=0)

    return inner


def preprocess_inceptionv1():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


def normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


STANDARD_TRANSFORMS: List[Callable[[T], T]] = [
    random_pad_image(10, padding_modes=["constant", "edge", "reflect"], fill=0.5),
    random_jitter_image(list(range(1, 10, 1))),
    random_scale_image([x / 100 for x in range(90, 120, 2)]),
    random_rotate_image(list(range(-10, 11, 1))),
]
