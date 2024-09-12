import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Literal, Optional, Tuple, Dict

from copy import deepcopy
import logging
from logging import Logger


M = nn.Module
T = torch.Tensor
A = np.ndarray


COLOUR_MAPPING = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "END": "\033[0m",
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


## loggers
LOG_LEVEL_MAPPING: Dict[str, int] = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_log_level_to_mine_logger(level: int | str) -> None:
    """Set the log level of the mine logger.

    Args:
        level: Logging level.
    """

    level = LOG_LEVEL_MAPPING[level.lower()] if isinstance(level, str) else level

    for _, logger in Logger.manager.loggerDict.items():
        if isinstance(logger, Logger):
            if hasattr(logger, "mine"):
                print(f"Setting log level of {logger.name} to {level}")
                logger.setLevel(level)


class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": COLOUR_MAPPING["CYAN"],
        "INFO": COLOUR_MAPPING["GREEN"],
        "VERBOSE": COLOUR_MAPPING["WHITE"],
        "WARNING": COLOUR_MAPPING["YELLOW"],
        "ERROR": COLOUR_MAPPING["RED"] + COLOUR_MAPPING["BOLD"],
        "CRITICAL": COLOUR_MAPPING["RED"]
        + COLOUR_MAPPING["BOLD"]
        + COLOUR_MAPPING["UNDERLINE"],
    }

    def format(self, record):
        formatted_record = deepcopy(record)

        level_name = formatted_record.levelname
        color = self.COLORS.get(level_name, "")

        # Adjust name based on whether it's in the main module or not
        formatted_record.name = (
            formatted_record.name
            if formatted_record.funcName == "<module>"
            else f"{formatted_record.name}.{formatted_record.funcName}"
        )

        # Create the log message without color first
        custom_format = "%(asctime)s - %(process)d - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        formatter = logging.Formatter(custom_format, datefmt="%Y-%m-%d %H:%M:%S")
        log_message = formatter.format(formatted_record)

        # Then apply color to the entire message
        colored_message = f"{color}{log_message}{COLOUR_MAPPING['END']}"

        return colored_message


def create_logger(
    name: str,
    level: Literal[
        "notset",
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ] = "info",
) -> Logger:
    """
    Create a logger with the specified name and level, including color formatting for console.

    Args:
        name: Name of the logger.
        level: Logging level. Defaults to "info".


    allowed_colours = ["black", "red", "green", "yellow", "blue", "cyan", "white",
                    "bold_black", "bold_red", "bold_green", "bold_yellow", "bold_blue",
                     "bold_cyan", "bold_white",
                    ]
    Returns:
        Configured logger object.
    """

    logger: Logger = logging.getLogger(name)
    logger.mine = True

    level_int: int = (
        LOG_LEVEL_MAPPING[level.lower()] if isinstance(level, str) else level
    )

    logger.setLevel(level_int)

    custom_formatter = CustomFormatter(
        "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # custom_formatter = logging.Formatter(
    #     "%(asctime)s - %(process)d - %(name)s: %(lineno)d - %(levelname)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with color formatting
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(level_int)
    console_handler.setFormatter(custom_formatter)
    logger.addHandler(console_handler)

    set_log_level_to_mine_logger(level_int)

    return logger


logger = create_logger(__name__)


def test_logger():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")






def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    elif activation == "log_softmax":
        return nn.LogSoftmax(dim=1)
    else:
        raise ValueError(f"Activation {activation} not supported")


def get_loss(loss: str):
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss == "mse":
        return nn.MSELoss()
    elif loss == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"Loss {loss} not supported")


def get_optimizer(optimizer: str, model: M, lr: float):
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
