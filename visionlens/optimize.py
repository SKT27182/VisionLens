import torch
import einops
import numpy as np

from typing import List, Tuple, Dict, Callable

from visionlens.objective import Objective, Hook
from visionlens.utils import T, M, A, AD, device, create_logger

logger = create_logger(__name__)

class Visualizer:

    def __init__(
        self,
        model: M,
        objective_f: Callable[[AD], float] | str,
        loss_type: str | Callable[[AD], float] = "mean",
    ):

        if isinstance(objective_f, str):
            self.objective_f = Objective.create_objective(objective_f, loss_type)

        else:
            self.objective_f = Objective(objective_f, "custom")

        self.model = model.to(device)

        model_hooks : Dict[str, Hook] = Hook.regester_hooks(self.model)

    def visualize(
        self,
        optimizer: torch.optim.Optimizer = None,
        transforms: List[Callable[[M], None]] = None,
        thresholds: Tuple[int] = (512,),
        verbose: bool = False,
        preprocess: bool = False,
        progress: bool = False,
        show_image: bool = True,
        save_image: bool = False,
        image_name: str | None = None,
        show_inline: bool = False,
        fixed_image_size: Tuple[int, int] | None = None,
    ):

        if optimizer is None:
            optimizer = torch.optim.Adam([self.model.parameters()], lr=5e-2)

        self.model.eval()

        if transforms is None:
            transforms = []
        else:
            transforms = transforms.copy()  # Avoid modifying the original list

        for threshold in thresholds:

            for transform in transforms:
                transform(self.model)

            if verbose:
                print(f"Threshold: {threshold}")

            for i in range(threshold):
                optimizer.zero_grad()
                self.model.zero_grad()

                loss = self.objective_f(self.model)

                loss.backward()
                optimizer.step()

                if verbose:
                    print(f"Loss: {loss.item()}")

            if show_image:
                self.show_image(
                    self.model,
                    preprocess,
                    show_inline,
                    fixed_image_size,
                    save_image,
                    image_name,
                )

            if progress:
                print(f"Threshold: {threshold}")
