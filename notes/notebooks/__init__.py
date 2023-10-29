import functools  # noqa: E402
import importlib  # noqa: E402
import itertools
import logging
import os  # noqa: E402
import warnings
from typing import Any, Optional  # noqa: E402

# Third Party
import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt
from scipy.optimize import minimize  # noqa: E402
from statsmodels import api as sm  # noqa: E402


def generic_plot(xlabel=None, ylabel=None, size=(10, 8)):
    _, ax = plt.figure(figsize=size)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _get_logger() -> Optional[logging.LoggerAdapter]:
    existing_logger = locals().get("logger", None)
    if existing_logger:
        existing_logger.warning("trying to overwrite logger; skipping")
        return
    app_logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s > %(message)s",
            datefmt="%d%m%y %I:%M:%S",
        )
    )
    app_logger.addHandler(handler)
    return logging.LoggerAdapter(app_logger, extra=None)


logger = _get_logger()
warnings.filterwarnings("ignore")


try:
    # Third Party
    from tqdm.notebook import tqdm, trange
except Exception as e:
    logger.warning(f"not in a notebook; using the standard tqdm: {e}")
    # Third Party
    from tqdm import tqdm, trange


try:
    # Third Party
    from IPython import get_ipython

    _ipy = get_ipython()
    _ipy.run_line_magic("load_ext", "watermark")
    _ipy.run_line_magic("watermark", "-n -u -v -iv -w")
    _ipy.run_line_magic("load_ext", "autoreload")
    _ipy.run_line_magic("autoreload", "2")
    del _ipy
except Exception as e:
    logger.warning(f"couldnt set up autoreload: {e}")
