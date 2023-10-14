# Third Party
# -*- coding: utf-8 -*-
# Third Party
import numba as nb
import numpy as np
import numpy.typing as npt

__author__ = "Petr Panov"
__copyright__ = "Copyleft 2023, Milky Way"
__license__ = "GNU"
__version__ = "1.0.0"
__email__ = "pvpanov93@gmail.com"
__status__ = "Draft"


@nb.njit(cache=True, nogil=True)
def cusums(arr: npt.NDArray[np.float_], whigh: float, wlow: float) -> tuple:
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
