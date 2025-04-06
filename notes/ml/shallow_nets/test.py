# %%
from statsmodels import api as sm
import importlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from jax import numpy as jnp

# %%
from notes.ml.shallow_nets import gen_data

df = importlib.reload(gen_data).generate_synthetic_timeseries(n_days=252 * 6)
# %%
train_df = df[:"2020-01-01"]
test_df = df["2020-01-01":]
X_train = train_df.iloc[:, 4:-3]
X_test = test_df.iloc[:, 4:-3]
y_train = train_df.iloc[:, -3:-1]
y_test = test_df.iloc[:, -3:-1]
w_train = train_df["weight"]
w_test = test_df["weight"]
# %%
f1 = sm.WLS(y_train.iloc[:, 0], sm.add_constant(X_train), w_train, missing="drop").fit()
f2 = sm.WLS(y_train.iloc[:, 1], sm.add_constant(X_train), w_train, missing="drop").fit()


# %%
def intersect_nonnan_data(*objects):
    valid_index_sets = []
    for obj in objects:
        valid_idx = (
            obj.dropna().index
        )  # For DataFrames: drops rows with any NaN; for Series: drops NaN values
        valid_index_sets.append(set(valid_idx))
    common_index = sorted(set.intersection(*valid_index_sets))
    return tuple(obj.loc[common_index] for obj in objects)


# %%
import pandas as pd
import numpy as np


def rmse(ytrue, yhat, w):
    return np.sqrt(np.sum(w[:, None] * (ytrue - yhat) ** 2))


ytrue_is, yhat_is, w_is = intersect_nonnan_data(
    y_train,
    pd.DataFrame(
        {
            "target1": f1.predict(sm.add_constant(X_train)),
            "target2": f2.predict(sm.add_constant(X_train)),
        }
    ),
    w_train,
)
ytrue_oos, yhat_oos, w_oos = intersect_nonnan_data(
    y_test,
    pd.DataFrame(
        {
            "target1": f1.predict(sm.add_constant(X_test)),
            "target2": f2.predict(sm.add_constant(X_test)),
        }
    ),
    w_test,
)

w_is_ = w_is.to_numpy() / w_is.sum()
w_oos_ = w_oos.to_numpy() / w_oos.sum()

print(f"RMSE on target 0 IS: ", rmse(ytrue_is, yhat_is, w_is_))
print(f"RMSE on target 0 OOS: ", rmse(ytrue_oos, yhat_oos, w_oos_))
# %%
np.sum(w_is_[:, None] * (ytrue_is - yhat_is) ** 2)
# %%
from notes.ml.shallow_nets import learn_data

trainer = importlib.reload(learn_data).WideDeepTrainer(
    X=jnp.array(X_train.to_numpy()),
    y=jnp.array(y_train.to_numpy()),
    weights=jnp.array(w_train.to_numpy()),
    wide_lr=0.1,
    deep_lr=1e-2,
    batch_size=64,
)
trainer.initial_training(n_initial_epochs=10)
# %%
r2_score(
    ytrue_is.to_numpy(),
    trainer.model(trainer.params, X_train.reindex(ytrue_is.index).to_numpy())[:, 0],
    sample_weight=w_is,
)
# %%
# %%
r2_score(
    ytrue_is.to_numpy(),
    yhat_is.to_numpy(),
    sample_weight=w_is,
)
# %%
plt.scatter(
    yhat_is.to_numpy(),
    trainer.model(trainer.params, X_train.loc[yhat_is.index].to_numpy())[:, 0],
)
# %%
trainer.standard_training(n_epochs=100, patience=5)
# %%
X_train
