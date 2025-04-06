# %%
import importlib
from matplotlib import pyplot as plt
import seaborn as sns

from notes.ml.shallow_nets import gen_data
# %%

df = generate_synthetic_timeseries(n_days=10)
print(df.head(10))
