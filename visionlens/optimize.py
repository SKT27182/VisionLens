import torch
import einops
import numpy as np

from typing import List, Tuple, Dict, Callable

from visionlens.utils import T, M, A


class Visualize:

    def __init__(
        self,
        model,
        objective_f,
        param_f=None,
    ):
        pass

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
