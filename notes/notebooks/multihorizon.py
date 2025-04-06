#%%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from numba import njit
import pandas as pd
#%%
ts_per_day = 25
wd_per_day = np.r_[[.5 / ts_per_day] * ts_per_day, .5]
n_days = 504
t0 = pd.Timestamp('2023-01-02')
t1 = t0 + pd.offsets.BDay(n_days)
dates = pd.bdate_range(t0, t1)
tds = pd.date_range(
    t0 + pd.Timedelta(hours=15, minutes=45),
    t0 + pd.Timedelta(hours=21, minutes=45),
    freq='15min',
) - t0
ts = pd.to_datetime((dates.to_numpy()[None, :] + tds.to_numpy()[:, None]).ravel())
wd = np.tile(wd_per_day, len(ts) // (ts_per_day + 1))
#%%
ts
#%%

#%%

#%%
t0 = pd.Timestamp('2024-01-03 15:45') + np.busday_offset([1, 2])
#%%
ts = pd.date_range(t0, '2024-01-03 21:45', freq='15min')
#%%
#%%
n_sim = 100_000
means = []
for _ in range(n_sim):
    lr_per_wd = np.random.randn(len(wd))
    lr = lr_per_wd * np.sqrt(wd) - wd / 2
    means.append(np.mean(np.exp(lr)))
#%%
import seaborn as sns
sns.kdeplot(means)