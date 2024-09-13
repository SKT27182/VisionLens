from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple, Dict, Callable, Union

from visionlens.utils import T, M, A
from decorator import decorator

from visionlens.utils import T, M, A, AD, device

# IMAGE_SHAPE -> (B, C, H, W)
# ACTIVATION_SHAPE -> (B, C, H, W)

class Hook:
    def __init__(self, module: M):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.features = None

    def hook_fn(self, module: M, input: T, output: T):
        self.features = output
        self.module = module

    def close(self):
        self.hook.remove()

    @staticmethod
    def close_all_hooks(model: M):
        for module in model.children():
            if len(list(module.children())) > 0:
                Hook.close_all_hooks(module)
            if hasattr(module, "forward_hooks"):
                for hook in module.forward_hooks:
                    hook.remove()
                module.forward_hooks = OrderedDict()
            if hasattr(module, "backward_hooks"):
                for hook in module.backward_hooks:
                    hook.remove()
                module.backward_hooks = OrderedDict()


class Objective:

    def __init__(self, objective_func: Callable[[AD], float], name: str):
        self.objective_func = objective_func
        self.name = name

    def __call__(self, act_dict: AD) -> float:
        return self.objective_func(act_dict)

    @staticmethod
    def create_objective(
        obj_str: str,
        loss_type: Union[str, Callable[[AD], float]] = "mean",
    ) -> "Objective":

        """
        obj_str: str, It contains the layer_name:channel:height:width, where channel, height, width are optional.

        loss_type: str | Callable[[T], float], The type of loss to be used. Default is "mean".

        Example:

            obj_str = "conv1:0:10:10" -> This will create an objective to maximize the activation of the 0th channel of the conv1 layer at the 10th row and 10th column
        """

        obj_str = obj_str.lower()

        obj_parts = obj_str.split(":")
        layer = obj_parts[0]
        channel = int(obj_parts[1]) if len(obj_parts) > 1 else None
        height = int(obj_parts[2]) if len(obj_parts) > 2 else None
        width = int(obj_parts[3]) if len(obj_parts) > 3 else None

        if channel is not None and (height is None or width is None):
            return neuron_obj(layer, channel, height, width, loss_type)

        elif channel is None:
            return layer_obj(layer, loss_type)

        else:
            return layer_obj(layer,  loss_type)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __add__(self, other: "Objective" | float | int) -> "Objective":
        if isinstance(other, Objective):

            objective_func = lambda act_dict: self(act_dict) + other(act_dict)
            name = f"{self.name}, {other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            objective_func = lambda act_dict: self(act_dict) + other
            name = f"{self.name} + {other}"
            return Objective(objective_func, name)

        else:

            raise ValueError(f"Unsupported type {type(other)} for addition")

    @staticmethod
    def sum(objectives: List["Objective"]) -> "Objective":
        objective_func = lambda act_dict: sum(
            [objective(act_dict) for objective in objectives]
        )
        name = ", ".join([objective.name for objective in objectives])
        return Objective(objective_func, name)

    def __sub__(self, other: "Objective" | float | int) -> "Objective":
        if isinstance(other, Objective):

            objective_func = lambda act_dict: self(act_dict) - other(act_dict)
            name = f"{self.name}, {other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            objective_func = lambda act_dict: self(act_dict) - other
            name = f"{self.name} - {other}"
            return Objective(objective_func, name)

        else:

            raise ValueError(f"Unsupported type {type(other)} for subtraction")

    def __neg__(self) -> "Objective":
        objective_func = lambda act_dict: -self(act_dict)
        name = f"-{self.name}"
        return Objective(objective_func, name)

    def __mul__(self, other: "Objective" | float | int) -> "Objective":
        if isinstance(other, Objective):

            objective_func = lambda act_dict: self(act_dict) * other(act_dict)
            name = f"{self.name}, {other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            objective_func = lambda act_dict: self(act_dict) * other
            name = f"{self.name} * {other}"
            return Objective(objective_func, name)

        else:

            raise ValueError(f"Unsupported type {type(other)} for multiplication")

    def __truediv__(self, other: "Objective" | float | int) -> "Objective":
        if isinstance(other, Objective):

            objective_func = lambda act_dict: self(act_dict) / other(act_dict)
            name = f"{self.name}, {other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            objective_func = lambda act_dict: self(act_dict) / other
            name = f"{self.name} / {other}"
            return Objective(objective_func, name)

        else:

            raise ValueError(f"Unsupported type {type(other)} for division")

    def __radd__(self, other: "Objective" | float | int) -> "Objective":
        return self.__add__(other)

    def __rsub__(self, other: "Objective" | float | int) -> "Objective":
        return self.__sub__(other)

    def __rmul__(self, other: "Objective" | float | int) -> "Objective":
        return self.__mul__(other)

    def __eq__(self, other: "Objective") -> bool:
        name_eq = self.name == other.name
        func_eq = self.objective_func == other.objective_func
        return name_eq and func_eq

    def __ne__(self, other: "Objective") -> bool:
        return not self.__eq__(other)


def activation_loss(
    actvations: T, loss_type: str | Callable[[T], float] = "mean" 
) -> float:
    
    loss_dict = {
        "mean": lambda x: x.mean(),
        "sum": lambda x: x.sum(),
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
    }

    if isinstance(loss_type, str):
        loss_fn = loss_dict[loss_type]
    else:
        loss_fn = loss_type

    return loss_fn(actvations)


###################### Objectives ######################

# objective_wrapper, decorator
def objective_wrapper(
    func: Callable[..., Callable[[AD], float]]
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        objective_func = func(*args, **kwargs)
        objective_name = func.__name__
        # args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        # description = objective_name.title() + args_str
        return Objective(objective_func, objective_name)

    return wrapper


@objective_wrapper
def channel_obj(layer, channel, loss_type="mean"):
    
    def get_activation_loss(act_dict):
        return -activation_loss(act_dict[layer][:, channel], loss_type)
    
    return get_activation_loss


@objective_wrapper
def layer_obj(layer, loss_type="mean"):
    
    def get_activation_loss(act_dict):
        return -activation_loss(act_dict[layer], loss_type)
    
    return get_activation_loss


@objective_wrapper
def neuron_obj(layer, channel, height=None, width=None, loss_type="mean"):

    def get_activation_loss(act_dict):

        if height is None and width is None:
            # Both height and width are None, default to half of the spatial dimensions
            height = act_dict[layer].shape[2] // 2
            width =  act_dict[layer].shape[3] // 2
            selected_activations = act_dict[layer][:, channel, height, width]

        elif height is not None and width is None:
            # Width is None, select the entire width dimension
            selected_activations = act_dict[layer][:, channel, height, :]

        elif height is None and width is not None:
            # Height is None, select the entire height dimension
            selected_activations = act_dict[layer][:, channel, :, width]

        return -activation_loss(selected_activations, loss_type)


    return get_activation_loss
