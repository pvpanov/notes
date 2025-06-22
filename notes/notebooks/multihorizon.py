# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from numba import njit
import pandas as pd

# %%
ts_per_day = 25
wd_per_day = np.r_[[0.5 / ts_per_day] * ts_per_day, 0.5]
n_days = 504
t0 = pd.Timestamp("2023-01-02")
t1 = t0 + pd.offsets.BDay(n_days)
dates = pd.bdate_range(t0, t1)
tds = (
    pd.date_range(
        t0 + pd.Timedelta(hours=15, minutes=45),
        t0 + pd.Timedelta(hours=21, minutes=45),
        freq="15min",
    )
    - t0
)
ts = pd.to_datetime((dates.to_numpy()[None, :] + tds.to_numpy()[:, None]).ravel())
wd = np.tile(wd_per_day, len(ts) // (ts_per_day + 1))
# %%
ts
# %%

# %%

# %%
t0 = pd.Timestamp("2024-01-03 15:45") + np.busday_offset([1, 2])
# %%
ts = pd.date_range(t0, "2024-01-03 21:45", freq="15min")
# %%
# %%
n_sim = 100_000
means = []
for _ in range(n_sim):
    lr_per_wd = np.random.randn(len(wd))
    lr = lr_per_wd * np.sqrt(wd) - wd / 2
    means.append(np.mean(np.exp(lr)))
# %%
import seaborn as sns

sns.kdeplot(means)
# %%
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# %%
xh = np.r_[-1.0, 3, 2, 0]
dxh = np.r_[3, 0, -1]


def loss(x):
    return np.sum((x - xh) ** 2) + np.sum((np.diff(x) - dxh) ** 2)


# %%
soln = minimize(loss, x0=xh).x
# %%
n = len(xh)
h = (
    2 * (2 * np.eye(n) + np.diag(np.r_[0, np.ones(n - 2), 0]))
    - np.eye(n, k=1)
    - np.eye(n, k=-1)
)
dx_dev = 2 * (np.diff(xh) - dxh)
g = -np.append(dx_dev, 0) + np.append(0, dx_dev)
plt.scatter(soln, xh - np.linalg.solve(h, g))
# %%
plt.plot(xh)
plt.plot(soln)
