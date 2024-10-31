import torch
import einops
import numpy as np
from IPython.display import clear_output

from tqdm.auto import tqdm

from typing import Callable, List, Literal, Optional, Tuple, Union

from visionlens.images import (
    compose,
    normalize,
    get_images,
    STANDARD_TRANSFORMS,
    preprocess_inceptionv1,
)
from visionlens.objectives import Objective, MultiHook, AD
from visionlens.utils import T, M, A, device, create_logger
from visionlens.img_utils import display_images_in_table, save_images

logger = create_logger(__name__)


class Visualizer:

    def __init__(
        self,
        model: M,
        objective_f: Callable[[AD], T] | str,
        model_hooks: Optional[Union[AD, MultiHook]] = None,
        loss_type: str | Callable[[T], T] = "mean",
        transforms: Union[List[Callable[[T], T]], None] = None,
        pre_process: bool = True,
    ):

        if isinstance(objective_f, str):
            logger.debug(f"Creating objective function from string: {objective_f}")
            self.objective_f = Objective.create_objective(objective_f, loss_type)

        else:
            logger.debug(f"Using custom objective function.")
            name = objective_f.name if hasattr(objective_f, "name") else "custom"
            self.objective_f = Objective(objective_f, name)

        self.model = model.to(device)

        if model_hooks is None:
            logger.debug("Creating model hooks from model")
            self.model_hooks = MultiHook(model)
        else:
            logger.debug("Using custom model hooks.")
            self.model_hooks = model_hooks

        if transforms is None:
            transforms = STANDARD_TRANSFORMS.copy()

        else:
            transforms = transforms.copy()

        if pre_process:
            if self.model._get_name() == "InceptionV1":
                transforms.append(preprocess_inceptionv1())
            else:
                transforms.append(normalize())

        self.transform_f = compose(transforms)

    def _single_forward_loop(
        self,
        model: M,
        optimizer: torch.optim.Optimizer,
        img_f: Callable[[], T],
    ) -> float:
        """Forward pass through the
        model and return the output tensor.

        Args:
            model: Model to run the forward pass.
            optimizer: Optimizer to update the parameters.
            img_f: Function to get the current image tensor.

        Returns:
            T: Output tensor.
        """

        model.eval()
        model.zero_grad()
        optimizer.zero_grad()

        model(self.transform_f(img_f()))

        loss = self.objective_f(self.model_hooks)

        logger.info(f"Loss: {loss.item():.2f}")

        # Compute gradients
        loss.backward()

        # Normalize the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the parameters
        optimizer.step()

        return loss.item()

    def visualize(
        self,
        param_f: Union[Callable[[], Tuple[List[T], Callable[[], T]]], None] = None,
        lr: float = 1e-1,
        freq: int = 10,
        threshold: Union[Tuple[int], List[int]] = (512,),
        use_decorrelated_img: bool = True,
        fft: bool = True,
        image_size: Union[
            int, Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (1, 3, 224, 224),
        save_path: str = "images/",
        save_images: bool = False,
        show_last: bool = True,
    ):
        """Visualize the model output by optimizing the input image.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 5e-2.
            freq (int, optional): Frequency to display the image. Defaults to 10.
            threshold (Tuple[int], optional): Epochs to save the image. Defaults to (512,).
            show_last (bool, optional): Show the final image. Defaults to True.
        """
        b, ch = 1, 3

        if isinstance(image_size, int):
            h, w = image_size, image_size
        elif len(image_size) == 2:
            h, w = image_size
        elif len(image_size) == 3:
            ch, h, w = image_size
        elif len(image_size) == 4:
            b, ch, h, w = image_size
        else:
            raise ValueError(
                "image_size must be an integer or a tuple of 2, 3, or 4 integers"
            )

        if param_f is None:
            logger.debug("Using random pixel image of shape (1, 3, 224, 224)")
            self.param_f = lambda: get_images(
                w=w,
                h=h,
                channels=ch,
                batch=b,
                device=device,
                fft=fft,
                decorrelate=use_decorrelated_img,
            )
        else:
            logger.debug("Using custom image parameter function")
            self.param_f = param_f

        params, img_f = self.param_f()

        batch_size = params[0].shape[0]

        optimizer = torch.optim.Adam(params, lr=lr)

        images: T = torch.empty(size=(0, *img_f().shape), device=device)

        losses: List[float] = []

        try:
            epochs = max(threshold) + 1
            for epoch in range(epochs):
                loss = self._single_forward_loop(self.model, optimizer, img_f)

                if epoch % freq == 0:
                    clear_output(wait=True)

                    display_images_in_table(img_f())

                    logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss:.2f}")

                if epoch in threshold:
                    im = img_f()
                    images = torch.cat(
                        (images, einops.rearrange(im, "b c h w -> 1 b c h w").detach())
                    )

                    if save_images:
                        logger.debug(f"Saving image at epoch {epoch}")
                        # add _{epoch} to the image name
                        save_images(im, save_path, f"image_{epoch}")

                losses.append(loss)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")

        if show_last:

            display_images_in_table(img_f())

        return images
