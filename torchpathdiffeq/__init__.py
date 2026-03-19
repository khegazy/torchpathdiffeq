from __future__ import annotations

import logging

from .base import IntegralOutput, steps
from .examples import ODE_dict, wolf_schlegel
from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .path_integral import ode_path_integral
from .runge_kutta import (
    RKParallelUniformAdaptiveStepsizeSolver,
    RKParallelVariableAdaptiveStepsizeSolver,
    get_parallel_RK_solver,
)
from .serial_solver import SerialAdaptiveStepsizeSolver


def setup_logging(level=logging.WARNING, filename=None):
    """Configure torchpathdiffeq logging.

    Sets up a handler on the ``torchpathdiffeq`` logger hierarchy.
    Call with ``logging.DEBUG`` to see all debug output.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
        filename: If provided, log to this file instead of stderr.
    """
    logger = logging.getLogger("torchpathdiffeq")
    logger.setLevel(level)
    handler = logging.FileHandler(filename) if filename else logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
