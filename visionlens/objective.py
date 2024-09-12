from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple, Dict, Callable

from visionlens.utils import T, M, A


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

    def __init__(self, objective_func: Callable[[Dict[str, T]], float], name: str):
        self.objective_func = objective_func
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, act_dict: Dict[str, T]) -> float:
        return self.objective_func(act_dict)

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
    actvations: T, loss_type: str | Callable[[T], float] = "mse" # type: ignore
) -> float:
    
    loss_dict = {
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy,
        "cross_entropy": F.cross_entropy,
    }

    if isinstance(loss_type, str):
        loss_func = loss_dict[loss_type]
    else:
        loss_func = loss_type

    return loss_func(actvations)