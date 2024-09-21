import torch
import einops
import numpy as np
from IPython.display import clear_output

from tqdm.auto import tqdm

from typing import Callable, List, Literal, Optional, Tuple, Union

from visionlens.images import (
    compose,
    normalize,
    pixel_image,
    STANDARD_TRANSFORMS,
    preprocess_inceptionv1,
)
from visionlens.objectives import Objective, MultiHook, AD
from visionlens.utils import T, M, A, device, create_logger
from visionlens.display_img_utils import display_images_in_table, save_image

logger = create_logger(__name__)


class Visualizer:

    def __init__(
        self,
        model: M,
        objective_f: Callable[[AD], T] | str,
        model_hooks: Optional[Union[AD, MultiHook]] = None,
        param_f: Union[Callable[[], Tuple[List[T], Callable[[], T]]], None] = None,
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

        if param_f is None:
            logger.debug("Using random pixel image of shape (1, 3, 224, 224)")
            param_f = lambda: pixel_image((1, 3, 224, 224))
        else:
            logger.debug("Using custom image parameter function")
            self.param_f = param_f

        if model_hooks is None:
            logger.debug("Creating model hooks from model")
            self.model_hooks = MultiHook(model)
        else:
            if isinstance(model_hooks, (MultiHook, AD)):
                self.model_hooks = model_hooks
            else:
                raise TypeError("model_hooks must be of type MultiHook")

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

        logger.info(f"Loss: {loss.item()}, type: {type(loss)}")

        loss.backward()

        optimizer.step()

        return loss.item()

    def _get_image_labels(self, batch_size: int, objective_name: str):

        objective_names = objective_name.split("_")

        objective_names = [name for name in objective_names if "diversity" not in name]

        n_objectives = len(objective_names)

        if n_objectives == batch_size:
            return objective_names
        elif n_objectives < batch_size:
            objective_name = objective_names[0]
            return [f"{objective_name}_{i}" for i in range(batch_size)]
        else:
            return objective_names[:batch_size]

    def visualize(
        self,
        lr: float = 1e-1,
        epochs: int = 200,
        freq: int = 10,
        threshold: Tuple[int] = (512,),
        save_images: Literal["last", "threshold", None] = None,
        save_path: str = "images/",
        show_last: bool = True,
    ):
        """Visualize the model output by optimizing the input image.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 5e-2.
            freq (int, optional): Frequency to display the image. Defaults to 10.
            threshold (Tuple[int], optional): Epochs to save the image. Defaults to (512,).
            show_last (bool, optional): Show the final image. Defaults to True.
            epochs (int, optional): Number of epochs to run. Defaults to 100.
        """

        params, img_f = self.param_f()

        batch_size = params[0].shape[0]
        
        if hasattr(self.objective_f, "name"):
            logger.debug(f"Objective function name: {self.objective_f.name}")
            images_labels = self._get_image_labels(batch_size, self.objective_f.name)
        else:
            logger.warning("Objective function does not have a name attribute.")
            images_labels = [f"image_{i}" for i in range(batch_size)]

        optimizer = torch.optim.Adam(params, lr=lr)

        images: T = torch.zeros((len(threshold), *img_f().shape), device=device)

        losses: List[float] = []

        try:
            for epoch in range(epochs):
                loss = self._single_forward_loop(self.model, optimizer, img_f)

                if epoch % freq == 0:
                    clear_output(wait=True)

                    display_images_in_table(img_f(), images_labels)

                    logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss}")

                if epoch in threshold:
                    im = img_f()
                    images[threshold.index(epoch)] = im

                    if save_images == "threshold":
                        saving_path = f"{save_path}/{self.objective_f.name}_{epoch}.png"
                        # save_image(im, saving_path)
                losses.append(loss)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")

        if show_last:

            display_images_in_table(img_f(), images_labels)

        return images
