import torch
import einops
import numpy as np
from IPython.display import clear_output

from tqdm.auto import tqdm

from typing import Callable

from visionlens.images import pixel_image
from visionlens.objective import Objective, MultiHook, AD
from visionlens.utils import T, M, A, device, create_logger
from visionlens.display_img_utils import display_images_in_table

logger = create_logger(__name__)


class Visualizer:

    def __init__(
        self,
        model: M,
        objective_f: Callable[[AD], float] | str,
        model_hooks: AD = None,
        param_f: Callable[[], T] = None,
        loss_type: str | Callable[[T], float] = "mean",
    ):

        if isinstance(objective_f, str):
            logger.debug(f"Creating objective function from string: {objective_f}")
            self.objective_f = Objective.create_objective(objective_f, loss_type)

        else:
            logger.debug(f"Using custom objective function.")
            self.objective_f = Objective(objective_f, "custom")

        self.model = model.to(device)

        if param_f is None:
            logger.debug("Using random pixel image of shape (1, 3, 224, 224)")
            param_f = lambda: pixel_image((1, 3, 224, 224))
        else:
            logger.debug("Using custom image parameter function")
            self.param_f = param_f

        if model_hooks is None:
            logger.debug("Creating model hooks from model")
            self.model_hooks: AD = MultiHook(model)
        else:
            self.model_hooks = model_hooks

    def _single_forward_loop(
        self,
        model: M,
        optimizer: torch.optim.Optimizer,
        img_f: Callable[[], T],
    ):
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

        model(img_f())

        loss = self.objective_f(self.model_hooks)

        loss.backward()

        optimizer.step()

        return loss.item()

    def visualize(
        self,
        lr: float = 5e-2,
        freq: int = 10,
        transforms: Callable[[T], T] = None,
        progress: bool = True,
        show_last: bool = True,
        epochs: int = 100,
    ):
        """Visualize the model output by optimizing the input image.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 5e-2.
            freq (int, optional): Frequency to display the image. Defaults to 10.
            progress (bool, optional): Show progress bar. Defaults to True.
            show_last (bool, optional): Show the final image. Defaults to True.
            epochs (int, optional): Number of epochs to run. Defaults to 100.
        """

        params, img_f = self.param_f()

        optimizer = torch.optim.Adam(params, lr=lr)

        losses = []
        with tqdm(total=epochs, disable=not progress, leave=True) as pbar:
            for epoch in range(epochs):
                loss = self._single_forward_loop(self.model, optimizer, img_f)

                if epoch % freq == 0:
                    clear_output(wait=True)
                    display_images_in_table([img_f()], ["Current Image"])

                    pbar.write(f"Epoch {epoch}/{epochs} - Loss: {loss}")

                losses.append(loss)
                pbar.update(1)

        if show_last:

            display_images_in_table([img_f()], ["Final Image"])
