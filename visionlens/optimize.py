import torch
import einops
import numpy as np

from typing import List, Tuple, Dict, Callable

from visionlens.objective import Objective
from visionlens.utils import T, M, A, AD, device


class Visualizer:

    def __init__(
        self,
        model: M,
        objective_f: Callable[[AD], float] | str,
        loss_type: str = "mean",
        param_f=None,
    ):

        if isinstance(objective_f, str):
            self.objective_f = Objective.create_objective(objective_f, loss_type)

        else:
            self.objective_f = Objective(objective_f, "custom")

        self.model = model.to(device)

        

    def visualize(
        self,
        optimizer=None,
        transforms=None,
        thresholds=(512,),
        verbose=False,
        preprocess=True,
        progress=True,
        show_image=True,
        save_image=False,
        image_name=None,
        show_inline=False,
        fixed_image_size=None,
    ): ...
