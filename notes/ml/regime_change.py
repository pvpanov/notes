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


def explicit_ridge(x, y, alpha, sample_weight=None):
    n, p = x.shape
    x = np.column_stack([x, np.ones_like(y)])
    xt = x.T / n
    if sample_weight is not None:
        sample_weight = sample_weight / sample_weight.mean()
        xt = x.T / sample_weight
    beta = np.linalg.solve(xt @ x + alpha * np.diag(np.r_[np.ones(p), 0.0]), xt @ y)
    std = (
        np.std(x @ beta - y)
        if sample_weight is None
        else np.std((x @ beta - y) * np.sqrt(sample_weight))
    )
    return beta[:-1], beta[-1], std


def bic_ridge_breaks(x, y, alpha, bps, sample_weight=None):
    """
    :param x: features
    :param y: univariate targets
    :param alpha: ridge penalty
    :param bps: breakpoint candidates
    :param sample_weight: if provided, sample weights
    :return: BIC change for each breakpoint
    """
    n, p = x.shape
    coefs, intercept, std = explicit_ridge(x, y, alpha, sample_weight=sample_weight)
    k_orig = p + 2
    bic_orig = k_orig * np.log(n) + 2 * n * np.log(std) + n * alpha * (coefs @ coefs)
    result = []
    k_split = k_orig + p + 3
    for bp in bps:
        (betal, interceptl, stdl), (betar, interceptr, stdr) = [
            explicit_ridge(
                x[sl],
                y[sl],
                alpha,
                sample_weight=None if sample_weight is None else sample_weight[sl],
            )
            for sl in [slice(None, bp), slice(bp, None)]
        ]
        nl = bp
        nr = n - bp
        bic_split = (
            k_split * np.log(n)
            + 2 * (nl * np.log(stdl) + nr * np.log(stdr))
            + alpha * (nl * (betal @ betal) + nr * (betar @ betar))
        )
        result.append(bic_split - bic_orig)
    return result


def bic_split(
    x,
    y,
    alpha: float,
    min_obs_before_break: int,
    nintervals_per_iteration: int = 16,
    sample_weight=None,
) -> list:
    """
    :param x: features
    :param y: univariate targets
    :param alpha: ridge penalty
    :param min_obs_before_break: minimum number of observations before/after a break
    :param nintervals_per_iteration: at each iteration we try `nintervals_per_iteration - 1` potential breaks
    :param sample_weight: if provided, sample weights
    :return: list of breakpoints
    """
    n = len(x)
    if n < 2 * min_obs_before_break:
        return []
    nintervals_per_iteration = min(nintervals_per_iteration, n // min_obs_before_break)
    bps = np.round(np.linspace(0, n, nintervals_per_iteration + 1)).astype(int)[1:-1]
    bics = bic_ridge_breaks(x, y, alpha, bps, sample_weight=None)
    ix_min = np.argmin(bics)
    if bics[ix_min] >= 0:
        return []
    bp = bps[ix_min]
    left_split, right_split = [
        bic_split(
            x[sl],
            y[sl],
            alpha,
            min_obs_before_break,
            nintervals_per_iteration,
            sample_weight=None if sample_weight is None else sample_weight[sl],
        )
        for sl in (slice(None, bp), slice(bp, None))
    ]
    return left_split + [bp] + [bp + j for j in right_split]


class RidgeWithChange:
    """we always fit the intercept"""

    def __init__(
        self, *, alpha: float, min_obs_before_break: int, nintervals_per_iteration: int
    ):
        self.alpha = alpha
        self.min_obs_before_break = min_obs_before_break
        self.nintervals_per_iteration = nintervals_per_iteration
        self.coef_, self.intercept_, self.scale_ = None, None, None

    def fit(self, x, y, *, sample_weight=None):
        if sample_weight is not None:
            sample_weight = sample_weight / sample_weight.sum()
        bps = bic_split(
            x,
            y,
            self.alpha,
            self.min_obs_before_break,
            self.nintervals_per_iteration,
            sample_weight,
        )
        n, p = x.shape
        if sample_weight is None:
            xm = np.mean(x, axis=0)
            xstd = np.std(x, axis=0)
        else:
            xm = x @ sample_weight
            xstd = np.sqrt(np.mean((x - xm) ** 2 @ sample_weight))
        xs = (x - xm) / xstd
        if len(bps) == 0:
            self.coef_, self.intercept_, self.scale_ = explicit_ridge(
                x, y, self.alpha, sample_weight
            )
            self.coef_ /= xstd
            self.intercept_ -= self.coef_ @ xm
            return self
        xs = np.column_stack([xs, np.ones(n, dtype=float)])
        x_with_dummy = np.column_stack(
            [xs[:, :-1]]
            + [
                xs
                * np.r_[
                    np.zeros(left_bp),
                    np.ones(right_bp - left_bp),
                    np.zeros(n - right_bp),
                ][:, None]
                for left_bp, right_bp in zip([0] + bps, bps + [n])
            ]
        )
        beta, intercept, _ = explicit_ridge(x_with_dummy, y, self.alpha, sample_weight)
        self.coef_ = (beta[:p] + beta[-1 - p : -1]) / xstd
        self.intercept_ = intercept + beta[-1] - self.coef_ @ xm
        last_slice = slice(bps[-1], n)
        last_resids = self.predict(x[last_slice]) - y[last_slice]
        if sample_weight is None:
            self.scale_ = np.std(last_resids)
        else:
            self.scale_ = np.sqrt((last_resids**2) @ sample_weight[last_slice])
        return self

    def predict(self, x):
        return x @ self.coef_ + self.intercept_

    def nll(self, x, y, sample_weight=None):
        elements = 2 * np.log(self.scale_) + (
            ((self.predict(x) - y) / self.scale_) ** 2
        )
        if sample_weight is None:
            return np.mean(elements)
        else:
            return (elements @ sample_weight) / sample_weight.sum()
