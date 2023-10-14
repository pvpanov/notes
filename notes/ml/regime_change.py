# Third Party
# -*- coding: utf-8 -*-
# Third Party
import numba as nb
import numpy as np
from numpy import typing as npt

__author__ = "Petr Panov"
__copyright__ = "Copyleft 2023, Milky Way"
__license__ = "GNU"
__version__ = "1.0.0"
__email__ = "pvpanov93@gmail.com"
__status__ = "Draft"


@nb.njit(cache=True, nogil=True)
def _cusums(arr: npt.NDArray[np.float_], whigh: float, wlow: float) -> tuple:
    n = len(arr)
    high_cusum = np.zeros(n, dtype=np.float_)
    low_cusum = np.zeros(n, dtype=np.float_)
    hc, lc = 0, 0
    for i, a in enumerate(arr):
        hc = max(0, hc + a - whigh)
        lc = max(0, lc - a - wlow)
        high_cusum[i] = hc
        low_cusum[i] = lc
    return high_cusum, low_cusum


def cusums(arr, whigh, wlow=None):
    return _cusums(arr, whigh, wlow or whigh)


@nb.njit(cache=True, nogil=True)
def _cusum_breakpoints(
    arr: npt.NDArray[np.float_],
    threshold_high: float,
    whigh: float,
    threshold_low: float,
    wlow: float,
    min_obs_to_break: int,
):
    bps = np.zeros(len(arr), dtype=np.int_)
    hc, lc = 0.0, 0.0
    steps_since_break, cur_loc = 0, 0
    for i, a in enumerate(arr):
        hc = max(0, hc + a - whigh)
        lc = max(0, lc - a - wlow)
        steps_since_break += 1
        if steps_since_break < min_obs_to_break:
            continue
        if hc > threshold_high or lc > threshold_low:
            lc, hc, steps_since_break, cur_loc, bps[cur_loc] = 0, 0, 0, cur_loc + 1, i
    return bps[:cur_loc]


def cusum_breakpoints(
    arr: npt.NDArray[np.float_],
    threshold_high: float,
    whigh: float,
    threshold_low=None,
    wlow=None,
    min_obs_to_break: int = 5,
):
    return _cusum_breakpoints(
        arr,
        threshold_high,
        whigh,
        threshold_low or threshold_high,
        wlow or whigh,
        min_obs_to_break,
    )
