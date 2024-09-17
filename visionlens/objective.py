from collections import OrderedDict
from functools import partial, wraps

from typing import Any, List, Dict, Callable, Tuple, Union

from visionlens.utils import T, M, create_logger

# IMAGE_SHAPE -> (B, C, H, W)
# ACTIVATION_SHAPE -> (B, C, H, W)

logger = create_logger(__name__)

AD = Callable[[str], "Hook"]  # Activation Dictionary Type for hooks [layer_name: Hook]


class Hook:

    def __init__(self, module: M):
        """
        Hook class to register forward hooks on the model.

        Args:
            module: M, The module on which the hook is to be registered.

        """
        logger.debug(f"Initializing Hook for module: {module.__class__.__name__}")
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module : M = module
        self.features : Union[T, None] = None

    def __call__(
        self,
    ):
        return self.features

    def hook_fn(self, module: M, input: T, output: T):
        """
        Hook function to be registered on the module.

        Args:
            module: M, The module on which the hook is registered.
            input: T, The input tensor to the module.
            output: T, The output tensor from the module.

        """
        logger.debug(f"Hook function called on module: {module}")
        self.features = output
        self.module = module

    def close(self):
        """
        Close the hook.
        """
        logger.debug(f"Closing hook for module: {self.module}")

        self.hook.remove()


class MultiHook:

    def __init__(
        self, model: M, layers: Union[List[str], str] = None, exact_match: bool = False
    ) -> AD:
        """
        MultiHook class to register

        Args:
            model: M, The model on which the hooks are to be registered.
            layers: List[str] | str, The list of layers on which the hooks are to be registered.
            exact_match: bool, Whether to match the layer names exactly.

        Returns:
            AD: The dictionary containing the layer name and the hook instance.
        """
        if isinstance(layers, str):
            layers = [layers]

        logger.info(
            f"Registering hooks for model: {model.__class__.__name__} with layers: {layers} and exact_match: {exact_match}"
        )
        self.hooks_dict = {}

        for name, module in model.named_modules():

            is_valid_layer = (
                any([layer in name for layer in layers]) if layers else False
            )

            if exact_match:
                is_valid_layer = (
                    any([layer == name for layer in layers]) if layers else False
                )

            if layers is None or is_valid_layer:

                logger.debug(f"Registering hook on layer {name}")

                self.hooks_dict[name] = Hook(module)

    def __call__(self, layer: str) -> T:
        """
        Get the features from the layer.

        Args:
            layer: str, The layer name from which the features are to be extracted.

        Returns:
            T: The features from the layer.
        """

        if layer == "labels":
            return list(self.hooks_dict.values())[-1]()
        else:
            assert (
                layer in self.hooks_dict
            ), f"Invalid layer {layer}. Get the list of layers using `visionlens.utils.get_model_layers(model)`"

            output = self.hooks_dict[layer]()

            assert (
                output is not None
            ), f"Features for layer {layer} not found. Make sure the model is in evaluation mode and at least one forward pass is made."

        return output

    def close(self):
        """
        Close all the hooks.
        """
        for hook in self.hooks_dict.values():
            hook.close()
        
    @staticmethod
    def close_all_hooks(model: M):
        """
        Close all the hooks on the model.

        Args:
            model: M, The model on which the hooks are to be closed.
        """
        logger.debug(f"Closing all hooks for model: {model.__class__.__name__}")
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

    def __init__(self, objective_func: Callable[[AD], T], name: str):
        logger.debug(f"Initializing Objective with name: {name}")
        self.objective_func = objective_func
        self.name = name

    def __call__(self, activation_dic: AD) -> T:
        logger.debug(f"Calling Objective: {self.name}")
        return self.objective_func(activation_dic)

    @staticmethod
    def create_objective(
        obj_str: str,
        loss_type: Union[str, Callable[[T], T]] = "mean",
    ) -> "Objective":
        """
        obj_str: str, It contains the layer_name:channel:height:width, where channel, height, width are optional.

        loss_type: str | Callable[[T], T], The type of loss to be used. Default is "mean".

        Example:

            obj_str = "conv1:0:10:10" -> This will create an objective to maximize the activation of the 0th channel of the conv1 layer at the 10th row and 10th column

        Returns:

                Objective: An instance of the Objective class.
        """
        logger.info(
            f"Creating Objective from string: {obj_str} with loss_type: {loss_type}"
        )
        obj_str = obj_str.lower()

        obj_parts = obj_str.split(":")
        layer = obj_parts[0]
        channel = int(obj_parts[1]) if len(obj_parts) > 1 else None
        height = int(obj_parts[2]) if len(obj_parts) > 2 else None
        width = int(obj_parts[3]) if len(obj_parts) > 3 else None

        if channel is not None and (height is not None or width is not None):
            logger.debug(
                f"Creating neuron objective for layer: {layer}, channel: {channel}, height: {height}, width: {width}, loss_type: {loss_type}"
            )
            return neuron_obj(layer, channel, height, width, loss_type, obj_str=obj_str)

        elif channel is not None:
            logger.debug(
                f"Creating channel objective for layer: {layer}, channel: {channel}, loss_type: {loss_type}"
            )
            return channel_obj(layer, channel, loss_type, obj_str=obj_str)

        else:
            logger.debug(
                f"Creating layer objective for layer: {layer}, loss_type: {loss_type}"
            )
            return layer_obj(layer, loss_type, obj_str=obj_str)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __add__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(f"Adding Objective: {self.name} with Objective: {other.name}")
            objective_func = lambda act_dict: self(act_dict) + other(act_dict)
            name = f"{self.name}_add_{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Adding Objective: {self.name} with value: {other}")
            objective_func = lambda act_dict: self(act_dict) + other
            name = f"{self.name}_add_{other}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for addition")

    @staticmethod
    def sum(objectives: Union[List["Objective"], Tuple["Objective", ...]]) -> "Objective":

        logger.debug(
            f"Summing Objectives: {[objective.name for objective in objectives]}"
        )
        objective_func = lambda act_dict: sum(
            [objective(act_dict) for objective in objectives]
        )
        name = " + ".join([objective.name for objective in objectives])
        return Objective(objective_func, name)

    def __sub__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(
                f"Subtracting Objective: {other.name} from Objective: {self.name}"
            )
            objective_func = lambda act_dict: self(act_dict) - other(act_dict)
            name = f"{self.name}_sub_{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Subtracting value: {other} from Objective: {self.name}")
            objective_func = lambda act_dict: self(act_dict) - other
            name = f"{self.name}_sub_{other}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for subtraction")

    def __neg__(self) -> "Objective":
        logger.debug(f"Negating Objective: {self.name}")
        objective_func = lambda act_dict: -self(act_dict)
        name = f"-{self.name}"
        return Objective(objective_func, name)

    def __mul__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(
                f"Multiplying Objective: {self.name} with Objective: {other.name}"
            )
            objective_func = lambda act_dict: self(act_dict) * other(act_dict)
            name = f"{self.name}_mul_{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Multiplying Objective: {self.name} with value: {other}")
            objective_func = lambda act_dict: self(act_dict) * other
            name = f"{other}{self.name}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for multiplication")

    def __truediv__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(f"Dividing Objective: {self.name} by Objective: {other.name}")
            objective_func = lambda act_dict: self(act_dict) / other(act_dict)
            name = f"{self.name}_div_{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Dividing Objective: {self.name} by value: {other}")
            objective_func = lambda act_dict: self(act_dict) / other
            name = f"{self.name}_div_{other}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for division")

    def __radd__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__add__(other)

    def __rsub__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__sub__(other)

    def __rmul__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__mul__(other)

    def __eq__(self, other: "Objective") -> bool:
        name_eq = self.name == other.name
        # func_eq = self.objective_func == other.objective_func
        return name_eq #and func_eq

    def __ne__(self, other: "Objective") -> bool:
        return not self.__eq__(other)


def activation_loss(
    actvations: T, loss_type: Union[str, Callable[[T], T]] = "mean"
) -> float:
    logger.debug(f"Calculating activation loss with loss_type: {loss_type}")
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
def objective_wrapper(func: Callable[..., Callable[[AD], float]]):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Wrapping function: {func.__name__}")
        objective_func = func(*args, **kwargs)
        # get obj_str from args
        objective_name = kwargs.get("obj_str", "")
        objective_name = objective_name if objective_name else func.__name__
        return Objective(objective_func, objective_name)

    return wrapper


@objective_wrapper
def channel_obj(layer, channel, loss_type="mean", obj_str=""):
    logger.info(
        f"Creating channel objective for layer: {layer}, channel: {channel}, loss_type: {loss_type}"
    )

    def get_activation_loss(act_dict):
        return -activation_loss(act_dict(layer)[:, channel], loss_type)

    return get_activation_loss


@objective_wrapper
def layer_obj(layer, loss_type="mean", obj_str=""):
    logger.info(f"Creating layer objective for layer: {layer}, loss_type: {loss_type}")

    def get_activation_loss(act_dict):
        return -activation_loss(act_dict(layer), loss_type)

    return get_activation_loss


@objective_wrapper
def neuron_obj(layer, channel, height=None, width=None, loss_type="mean", obj_str=""):
    logger.info(
        f"Creating neuron objective for layer: {layer}, channel: {channel}, height: {height}, width: {width}, loss_type: {loss_type}"
    )

    def get_activation_loss(act_dict):
        if height is None and width is None:
            # Both height and width are None, default to half of the spatial dimensions
            height = act_dict(layer).shape[2] // 2
            width = act_dict(layer).shape[3] // 2
            selected_activations = act_dict(layer)[:, channel, height, width]

        elif height is not None and width is None:
            # Width is None, select the entire width dimension
            selected_activations = act_dict(layer)[:, channel, height, :]

        elif height is None and width is not None:
            # Height is None, select the entire height dimension
            selected_activations = act_dict(layer)[:, channel, :, width]

        return -activation_loss(selected_activations, loss_type)

    return get_activation_loss
