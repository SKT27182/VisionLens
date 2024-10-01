from collections import OrderedDict
from functools import partial, wraps

from typing import Any, List, Dict, Callable, Tuple, Union

import einops
import torch
import torch.nn.functional as F

from visionlens.utils import T, M, create_logger, get_nth_batch

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
        self.module: M = module
        self.features: Union[T, None] = None

    def __call__(
        self,
    ):
        return self.features

    def hook_fn(self, module: M, input: Tuple[Any, ...], output: Any):
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
        self,
        model: M,
        layers: Union[List[str], str, None] = None,
        exact_match: bool = False,
    ):
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
                MultiHook.close_all_hooks(module)
            if hasattr(module, "forward_hooks"):
                if isinstance(module.forward_hooks, OrderedDict):
                    for hook in module.forward_hooks:
                        hook.remove()
                # module.forward_hooks = OrderedDict()
            if hasattr(module, "backward_hooks"):
                if isinstance(module.backward_hooks, OrderedDict):
                    for hook in module.backward_hooks:
                        hook.remove()
                # module.backward_hooks = OrderedDict()


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
        obj_name: str, loss_type: Union[str, Callable[[T], T]] = "mean", batch=None
    ) -> "Objective":
        """
        obj_name: str, It contains the layer_name:channel:height:width, where channel, height, width are optional.

        loss_type: str | Callable[[T], T], The type of loss to be used. Default is "mean".

        Example:

            obj_name = "conv1:0:10:10" -> This will create an objective to maximize the activation of the 0th channel of the conv1 layer at the 10th row and 10th column

        Returns:

                Objective: An instance of the Objective class.
        """
        logger.info(
            f"Creating Objective from string: {obj_name} with loss_type: {loss_type}"
        )
        obj_name = obj_name.lower()

        obj_parts = obj_name.split(":")
        layer = obj_parts[0]
        channel = (
            int(obj_parts[1]) if (len(obj_parts) > 1) & (obj_parts[1] != "") else None
        )
        height = (
            int(obj_parts[2]) if (len(obj_parts) > 2) & (obj_parts[2] != "") else None
        )
        width = (
            int(obj_parts[3]) if (len(obj_parts) > 3) & (obj_parts[3] != "") else None
        )

        obj_name = f"{obj_name}-{batch}" if batch is not None else obj_name

        if channel is not None and (height is not None or width is not None):
            return neuron_obj(
                layer,
                channel=channel,
                height=height,
                width=width,
                loss_type=loss_type,
                obj_name=obj_name,
            )

        elif channel is not None:
            return channel_obj(
                layer, channel, loss_type, obj_name=obj_name, batch=batch
            )

        else:
            return layer_obj(layer, loss_type, obj_name=obj_name, batch=batch)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __add__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(f"Adding Objective: {self.name} with Objective: {other.name}")
            objective_func = lambda act_dict: self(act_dict) + other(act_dict)
            name = f"{self.name}+{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Adding Objective: {self.name} with value: {other}")
            objective_func = lambda act_dict: self(act_dict) + other
            name = f"{self.name}+{other}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for addition")

    @staticmethod
    def sum(
        objectives: Union[List["Objective"], Tuple["Objective", ...]]
    ) -> "Objective":

        logger.debug(
            f"Summing Objectives: {[objective.name for objective in objectives]}"
        )
        objective_func = lambda act_dict: sum(
            [objective(act_dict) for objective in objectives]
        )
        name = "+".join([objective.name for objective in objectives])
        return Objective(objective_func, name)

    def __sub__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(
                f"Subtracting Objective: {other.name} from Objective: {self.name}"
            )
            objective_func = lambda act_dict: self(act_dict) - other(act_dict)
            name = f"{self.name}-{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Subtracting value: {other} from Objective: {self.name}")
            objective_func = lambda act_dict: self(act_dict) - other
            name = f"{self.name}-{other}"
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
            name = f"{self.name}*{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Multiplying Objective: {self.name} with value: {other}")
            objective_func = lambda act_dict: self(act_dict) * other
            name = f"{other}*{self.name}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for multiplication")

    def __truediv__(self, other: Union["Objective", float, int]) -> "Objective":
        if isinstance(other, Objective):
            logger.debug(f"Dividing Objective: {self.name} by Objective: {other.name}")
            objective_func = lambda act_dict: self(act_dict) / other(act_dict)
            name = f"{self.name}/{other.name}"
            return Objective(objective_func, name)

        elif isinstance(other, (float, int)):
            logger.debug(f"Dividing Objective: {self.name} by value: {other}")
            objective_func = lambda act_dict: self(act_dict) / other
            name = f"{self.name}/{other}"
            return Objective(objective_func, name)

        else:
            raise ValueError(f"Unsupported type {type(other)} for division")

    def __radd__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__add__(other)

    def __rsub__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__sub__(other)

    def __rmul__(self, other: Union["Objective", float, int]) -> "Objective":
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Objective):
            return False
        name_eq = self.name == other.name
        # func_eq = self.objective_func == other.objective_func
        return name_eq  # and func_eq

    def __ne__(self, other: object) -> bool:
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
def _extract_act_pos(acts, x=None, y=None):
    shape = acts.shape

    if y is None and x is None:
        # Both y and x are None, default to half of the spatial dimensions
        y_ = acts.shape[2] // 2
        x_ = acts.shape[3] // 2
        logger.debug(f"Using default y: {y_} and x: {x_}")
        selected_activations = acts[..., y_ : y_ + 1, x_ : x_ + 1]

    elif y is not None and x is None:
        # Width is None, select the entire x dimension
        logger.debug(f"Selecting entire x dimension for y: {y }")
        selected_activations = acts[..., y : y + 1, :]

    elif y is None and x is not None:
        # Height is None, select the entire y dimension
        logger.debug(f"Selecting entire y dimension for x: {x}")
        selected_activations = acts[..., :, x : x + 1]

    else:
        # Both y and x are not None
        logger.debug(f"Selecting activation at y: {y} and x: {x}")
        selected_activations = acts[..., y : y + 1, x : x + 1]

    # x = shape[2] // 2 if x is None else x
    # y = shape[3] // 2 if y is None else y
    return selected_activations


def handle_batch(batch=None):
    """
    A decorator to handle batch processing for a given function.

    Parameters:
    batch (int, optional): The batch number to process. If None, the default behavior is applied.

    Returns:
    function: A decorator that wraps the given function to process the specified batch.
    """

    def decorator(get_activation_loss):
        """
        A decorator to wrap a function that computes activation loss.
        Args:
            get_activation_loss (function): A function that computes the activation loss.
        Returns:
            function: A wrapped function that takes an activation dictionary and computes the loss for the nth batch.
        """

        def wrapper(act_dict):
            nth_batch_act = get_nth_batch(act_dict, batch)
            return get_activation_loss(nth_batch_act)

        return wrapper

    return decorator


# objective_wrapper, decorator
def objective_wrapper(
    func: Callable[..., Callable[[AD], float]],
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Wrapping function: {func.__name__}")
        objective_func, objective_name = func(*args, **kwargs)
        # get obj_name from args
        # objective_name = kwargs.get("obj_name", func.__name__)
        return Objective(objective_func, objective_name)

    return wrapper


# objectives
@objective_wrapper
def channel_obj(layer, channel, loss_type="mean", obj_name="", batch=None):
    logger.info(
        f"Creating channel objective for layer: {layer}, channel: {channel}, loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = f"{layer}:{channel}" if not obj_name else obj_name

    @handle_batch(batch)
    def get_activation_loss(act_dict):
        return -activation_loss(act_dict(layer)[:, channel], loss_type)

    return get_activation_loss, obj_name


@objective_wrapper
def layer_obj(layer, loss_type="mean", obj_name="", batch=None):
    logger.info(
        f"Creating layer objective for layer: {layer}, loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = f"{layer}" if not obj_name else obj_name

    @handle_batch(batch)
    def get_activation_loss(act_dict):
        return -activation_loss(act_dict(layer), loss_type)

    return get_activation_loss, obj_name


@objective_wrapper
def neuron_obj(
    layer, channel, height, width, loss_type="mean", obj_name="", batch=None
):
    logger.info(
        f"Creating neuron objective for layer: {layer}, channel: {channel}, height: {height}, width: {width}, loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = f"{layer}:{channel}:{height}:{width}" if not obj_name else obj_name

    @handle_batch(batch)
    def get_activation_loss(act_dict):

        # if height is None and width is None:
        #     # Both height and width are None, default to half of the spatial dimensions
        #     height_ = act_dict(layer).shape[2] // 2
        #     width_ = act_dict(layer).shape[3] // 2
        #     logger.debug(f"Using default height: {height_} and width: {width_}")
        #     selected_activations = act_dict(layer)[:, channel, height_, width_]

        # elif height is not None and width is None:
        #     # Width is None, select the entire width dimension
        #     logger.debug(f"Selecting entire width dimension for height: {height }")
        #     selected_activations = act_dict(layer)[:, channel, height, :]

        # elif height is None and width is not None:
        #     # Height is None, select the entire height dimension
        #     logger.debug(f"Selecting entire height dimension for width: {width}")
        #     selected_activations = act_dict(layer)[:, channel, :, width]

        # else:
        #     # Both height and width are not None
        #     logger.debug(f"Selecting activation at height: {height} and width: {width}")
        #     selected_activations = act_dict(layer)[:, channel, height, width]
        selected_activations = _extract_act_pos(
            act_dict(layer)[:, channel, :, :], x=width, y=height
        )

        return -activation_loss(selected_activations, loss_type)

    return get_activation_loss, obj_name


@objective_wrapper
def diversity(layer, batch=None):
    logger.info(
        f"Creating diversity objective for layer: {layer}, batch: {batch if batch else 'all'}"
    )

    obj_name = f"{layer}:diversity"

    def get_activation_loss(act_dict):
        activations = act_dict(layer)

        if isinstance(batch, int):
            activations = activations[batch : batch + 1]
        elif batch is not None:
            activations = activations[batch]

        batch_size, channels, _, _ = activations.shape

        flattened_activations = einops.rearrange(activations, "b c h w -> b c (h w)")
        # Calculate the cosine similarity between the activations
        grams = torch.einsum(
            "bcs,bds->bcd", flattened_activations, flattened_activations
        )

        # First we flatten the activations -> (b, c, s), where s -> h * w
        # then we calculate the cosine similarity between the activations by taking the dot product
        # of the flattened activations -> (b, c, s) . (b, s, c) -> (b, c, c)

        # Normalize the gram matrix, across the channel dimension,
        grams = F.normalize(grams, p=2, dim=(1, 2))

        # Calculate the diversity loss, maximizing the diversity between the different t batches
        # diversity_loss = (
        #     -sum(
        #         [
        #             sum(
        #                 [
        #                     (grams[i] * grams[j]).sum()
        #                     for j in range(batch_size)
        #                     if j != i
        #                 ]
        #             )
        #             for i in range(batch_size)
        #         ]
        #     )
        #     / batch_size
        # )

        # Create a mask to ignore cases where m == n
        mask = torch.eye(batch_size, device=grams.device).bool()
        # maximize the diversity between the different the batches
        diversity_loss = (
            -torch.einsum("mij,mij->m", grams, grams).masked_fill(mask, 0).sum()
            / batch_size
        )

        return diversity_loss

    return get_activation_loss, obj_name


@objective_wrapper
def direction_in_layer_obj(layer, direction, loss_type="mean", obj_name="", batch=None):
    """
    Create an objective to maximize the activation in the direction in the layer.
    - Basis for the direction is the mean activation in each channel.

    Args:
        layer: str, The layer name.
        direction: (A,T), The direction in which the activation is to be maximized. (C,)
        loss_type: str, The type of loss to be used. Default is "mean".
        obj_name: str, The name of the objective. Default is "".
        batch: int, The batch number to process. Default is None.

    Returns:
        Callable[[AD], float]: The objective function.

    """

    logger.info(
        f"Creating direction in layer objective for layer: {layer},  loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = f"{layer}:direction:{direction}" if not obj_name else obj_name

    @handle_batch(batch)
    def get_activation_loss(act_dict):
        activations = act_dict(layer)
        # Calculate the mean activation in each channel
        # mean_activations = activations.mean(
        #     dim=(2, 3)
        # )  # this is the basis for the direction
        # Calculate the dot product between the mean activations and the direction
        direction_in_layer = torch.nn.CosineSimilarity(dim=1)(
            direction.reshape(1, -1, 1, 1), activations
        )

        logger.debug(
            f"layer shape: {activations.shape}, direction shape: {direction.shape}, cosine_similarity shape: {direction_in_layer.shape}"
        )

        return -activation_loss(direction_in_layer, loss_type)

    return get_activation_loss, obj_name


@objective_wrapper
def direction_at_position(
    layer, direction, x=None, y=None, loss_type="mean", obj_name="", batch=None
):
    """
    Create an objective to maximize the activation in the direction at the position in the layer.

    Args:
        layer: str, The layer name.
        direction: (A,T), The direction in which the activation is to be maximized. (C,)
        x: int, The x-coordinate of the position. Default is None.
        y: int, The y-coordinate of the position. Default is None.
        loss_type: str, The type of loss to be used. Default is "mean".
        obj_name: str, The name of the objective. Default is "".
        batch: int, The batch number to process. Default is None.

    Returns:
        Callable[[AD], float]: The objective function.

    """

    logger.info(
        f"Creating direction at position objective for layer: {layer}, x: {x}, y: {y}, loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = (
        f"{layer}:direction:{direction}:position:{x}:{y}" if not obj_name else obj_name
    )

    @handle_batch(batch)
    def get_activation_loss(act_dict):
        activations = act_dict(layer)
        # Calculate the dot product between the direction and the activations at the position

        activations = _extract_act_pos(activations, x=x, y=y)
        print(activations.shape)

        direction_at_position = torch.nn.CosineSimilarity(dim=1)(
            direction.reshape(1, -1, 1, 1), activations
        )

        logger.debug(
            f"layer shape: {activations.shape}, direction shape: {direction.shape}, cosine_similarity shape: {direction_at_position.shape}"
        )

        return -activation_loss(direction_at_position, loss_type)

    return get_activation_loss, obj_name


@objective_wrapper
def channel_interpolate(
    layer1, channel1, layer2, channel2, alpha, loss_type="mean", obj_name="", batch=None
):
    """
    Create an objective to interpolate between the activations of two channels in two layers.

    Args:
        layer1: str, The layer name of the first channel.
        channel1: int, The channel number of the first channel.
        layer2: str, The layer name of the second channel.
        channel2: int, The channel number of the second channel.
        alpha: float, The interpolation factor.
        loss_type: str, The type of loss to be used. Default is "mean".
        obj_name: str, The name of the objective. Default is "".
        batch: int, The batch number to process. Default is None.

    Returns:
        Callable[[AD], float]: The objective function.

    """

    logger.info(
        f"Creating channel interpolate objective for layer1: {layer1}, channel1: {channel1}, layer2: {layer2}, channel2: {channel2}, alpha: {alpha}, loss_type: {loss_type}, batch: {batch}"
    )

    obj_name = (
        f"{layer1}:{channel1}:interpolate:{layer2}:{channel2}:{alpha}"
        if not obj_name
        else obj_name
    )

    @handle_batch(batch)
    def get_activation_loss(act_dict):
        activations1 = act_dict(layer1)
        activations2 = act_dict(layer2)
        # Calculate the interpolation between the activations of the two channels
        # interpolated_activations = (
        #     alpha * activations1[:, channel1] + (1 - alpha) * activations2[:, channel2]
        # )

        for n in range(activations1.shape[0]):
            interpolated_activations = (
                alpha * activations1[n, channel1]
                + (1 - alpha) * activations2[n, channel2]
            )

        return -activation_loss(interpolated_activations, loss_type)

    return get_activation_loss, obj_name
