import torch
import einops
import numpy as np

from visionlens.utils import device, create_logger

logger = create_logger(__name__)

color_correlation_svd_sqrt = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()  # (B, C, H, W)
    image = einops.rearrange(image, "b c h w -> b h w c")
    # Check if the image is single channel and convert to 3-channel
    if len(image.shape) == 4 and image.shape[3] == 1:  # Single channel image
        image = einops.repeat(image, "b h w 1 -> b h w c", c=3)
    return image


def _linear_decorrelate_color(tensor):
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(
        t_permute, torch.tensor(color_correlation_normalized.T).to(t_permute.device)
    )
    tensor = t_permute.permute(0, 3, 1, 2)
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


def image(
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
