import pandas as pd
from dataclasses import dataclass, field
from functools import partial
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random, jit, value_and_grad
from jax.nn.initializers import glorot_uniform


@dataclass
class WideDeepTrainer:
    data: pd.DataFrame
    wide_cols: tuple[str, ...]
    deep_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    weight_col: str
    n_groups: int

    wide_lr: float = 0.1
    deep_lr: float = 0.01
    n_epochs: int = 10

    lambda_wide_l2: float = 1e-3
    lambda_wide_l1: float = 1e-3
    lambda_deep_l2: float = 1e-3
    lambda_deep_l1: float = 1e-3
    ridge_lambda: float = 1e-2

    deep1_size: int = 8
    deep2_size: int = 4

    seed: int = 42

    params: dict = field(init=False)
    opt_state: Any = field(init=False)
    tx: Any = field(init=False)
    key: Any = field(init=False)

    def __post_init__(self):
        self.key = random.PRNGKey(self.seed)
        self.x_wide = self.data[self.wide_cols].to_numpy()
        self.x_deep = self.data[self.deep_cols].to_numpy()
        self.y = self.data[self.target_cols].to_numpy()
        self.w = self.data[self.weight_col].to_numpy()
        W_wide, b_wide = self.ridge_regression(
            np.array(self.X),
            np.array(self.y),
            np.array(self.weights),
            self.ridge_lambda,
        )

        # Initialize deep branch parameters using Glorot uniform.
        self.key, subkey1, subkey2 = random.split(self.key, 3)
        W1 = self.glorot_uniform(subkey1, (d_in, self.deep_layer1_size))
        b1 = jnp.zeros((self.deep_layer1_size,))
        W2 = self.glorot_uniform(
            subkey2, (self.deep_layer1_size, self.deep_layer2_size)
        )
        b2 = jnp.zeros((self.deep_layer2_size,))

        # Combine into a parameter pytree.
        self.params = {
            "wide": {"W": jnp.array(W_wide), "b": jnp.array(b_wide)},
            "deep": {"W1": W1, "b1": b1, "W2": W2, "b2": b2},
        }

        # Create a pytree of parameter labels for multi-transform.
        param_labels = {
            "wide": {"W": "wide", "b": "wide"},
            "deep": {"W1": "deep", "b1": "deep", "W2": "deep", "b2": "deep"},
        }
        # Define the optimizer with separate learning rates for wide and deep parts.
        self.tx = optax.multi_transform(
            {"wide": optax.adam(self.wide_lr), "deep": optax.adam(self.deep_lr)},
            param_labels,
        )
        self.opt_state = self.tx.init(self.params)

    @staticmethod
    def _ridge(x, y, w, lam):
        """Solve for ridge regression parameters (W, b) using an augmented design matrix."""
        mask = self.mask_from_triple(X, y, w)
        X, y, w = (z[mask] for z in (X, y, w))
        n, d = X.shape
        X_aug = np.concatenate([X, np.ones((n, 1))], axis=1)
        I = np.eye(d + 1)
        I[-1, -1] = 0  # Do not regularize the intercept.
        xtw = X_aug.T * (w / w.sum())
        theta = np.linalg.solve(xtw @ X_aug + lam * I, xtw @ y)
        W = theta[:-1, :]
        b = theta[-1, :]
        return W.astype(np.float32), b.astype(np.float32)

    def glorot_uniform(self, key, shape):
        fan_in, fan_out = shape[0], shape[1]
        limit = jnp.sqrt(6 / (fan_in + fan_out))
        return random.uniform(key, shape, minval=-limit, maxval=limit)

    def leaky_relu(self, x, alpha=0.01):
        return jnp.where(x > 0, x, alpha * x)

    def model(self, params, x):
        """Compute the output of the wide & deep model."""
        # Wide branch: a linear layer.
        wide_out = jnp.dot(x, params["wide"]["W"]) + params["wide"]["b"]
        # Deep branch: two-layer MLP with leaky ReLU activations.
        h1 = self.leaky_relu(jnp.dot(x, params["deep"]["W1"]) + params["deep"]["b1"])
        h2 = self.leaky_relu(jnp.dot(h1, params["deep"]["W2"]) + params["deep"]["b2"])
        return wide_out + h2

    def loss_fn(self, params, batch):
        # Unpack batch; if boundaries are provided, batch is a 4-tuple.
        if len(batch) == 4:
            x, y, w, boundaries = batch
        else:
            x, y, w = batch
            boundaries = None

        yhat = self.model(params, x)  # shape: (batch, num_groups)

        # Compute a valid mask per row (only rows where all entries in y, yhat, and w are finite)
        valid_mask = jnp.all(jnp.isfinite(yhat), axis=1) & jnp.all(jnp.isfinite(y), axis=1) & jnp.isfinite(w)
        valid_mask_f = valid_mask.astype(yhat.dtype)

        # --- Term 1: Negative PnL ---
        # Compute weighted PnL only over valid rows.
        w_exp = w[:, None]
        pnl = jnp.sum(w_exp * (y * yhat) * valid_mask_f[:, None]) / jnp.sum(w * valid_mask_f)
        loss1 = -pnl

        # --- Term 2: Variance penalty ---
        weighted_sum = jnp.sum(y * yhat, axis=1)
        loss2 = self.gamma * (jnp.sum(w * (weighted_sum ** 2) * valid_mask_f) / jnp.sum(w * valid_mask_f))

        # --- Term 3: Transaction cost penalty ---
        # First, reshape yhat into (T, num_groups), where T = number of timestamps in the batch.
        T_batch = yhat.shape[0] // self.num_groups
        yhat_grouped = yhat.reshape((T_batch, self.num_groups))
        
        # Define a helper to compute transaction cost for a given block.
        def compute_block_cost(block):
            # block is shape (block_T, num_groups)
            valid_block = jnp.all(jnp.isfinite(block), axis=1)
            diff = jnp.abs(block[1:] - block[:-1])
            valid_diff = valid_block[:-1] & valid_block[1:]
            # For each group, sum only differences where both adjacent rows are valid.
            return jnp.sum(self.c * jnp.sum(diff * valid_diff[:, None], axis=0))
        
        if boundaries is None:
            # If no boundaries provided, compute transaction cost over the whole batch.
            valid_mask_grouped = jnp.all(jnp.isfinite(yhat_grouped), axis=1)
            diff = jnp.abs(yhat_grouped[1:] - yhat_grouped[:-1])
            valid_diff = valid_mask_grouped[:-1] & valid_mask_grouped[1:]
            loss3 = jnp.sum(self.c * jnp.sum(diff * valid_diff[:, None], axis=0))
        else:
            # boundaries is a 1D array of indices (length = num_blocks+1) that partition the batch
            # into contiguous blocks (substrides). We compute transaction cost per block.
            def block_cost(i, acc):
                start = boundaries[i]
                end = boundaries[i + 1]
                block = yhat[start:end, :].reshape((-1, self.num_groups))
                cost_block = compute_block_cost(block)
                return acc + cost_block, None

            num_blocks = boundaries.shape[0] - 1
            loss3, _ = jax.lax.scan(block_cost, 0.0, jnp.arange(num_blocks))

        return loss1 + loss2 + loss3

    @staticmethod
    @partial(jit, static_argnums=(3, 4))
    def update_step_fn(params, opt_state, batch, tx, loss_fn):
        loss_val, grads = value_and_grad(loss_fn)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    def update_step(self, params, opt_state, batch):
        X, y, w = batch
        mask = self.mask_from_triple(X, y, w)
        batch = (X[mask], y[mask], w[mask])
        return WideDeepTrainer.update_step_fn(
            params, opt_state, batch, self.tx, self.loss_fn
        )

    def striped_batches_grouped(self, X, y, weights):
        """
        Constructs striped mini-batches with equal representation from groups A and B,
        using a pattern based on timestamps. Assumes:
        - X has shape (n_samples, d), where n_samples = 2*T (T timestamps, each with 2 samples).
        - T is a perfect cube.
        For T timestamps, let s = T^(1/3). Then, for each split j in 0,...,s-1, we select, for each block k,
        a contiguous block of s timestamps starting at index = k*period + j*s, where period = s^2.
        """
        # Number of timestamps
        T = X.shape[0] // 2
        s = int(round(T ** (1 / 3)))  # number of splits and block size
        block_size = s
        period = s * s
        num_blocks = T // period  # ideally, T == s^3 so num_blocks == s

        # For each split j, collect sample indices.
        for j in range(s):
            indices = []
            for k in range(num_blocks):
                start = k * period + j * block_size
                for t in range(start, start + block_size):
                    # Convert timestamp index t to sample indices (2 per timestamp).
                    indices.extend([t * 2, t * 2 + 1])
            indices = np.array(indices)
            batch_x = X[indices, :]
            batch_y = y[indices, :]
            batch_w = weights[indices]
            yield batch_x, batch_y, batch_w

    def get_batches(self, x, y, w, batch_size):
        n = x.shape[0]
        for i in range(0, n, batch_size):
            yield x[i : i + batch_size], y[i : i + batch_size], w[i : i + batch_size]

    def rmse(self, params, x, y, w):
        preds = self.model(params, x)
        return jnp.sqrt(WideDeepTrainer.nonnan_mse(preds, y, w))

    def initial_training(self, n_initial_epochs=3):
        """Initial phase: train using striped mini-batches (timestamp-based) ensuring equal A/B representation."""
        for epoch in range(n_initial_epochs):
            for batch in self.striped_batches_grouped(self.X, self.y, self.weights):
                self.params, self.opt_state, loss_val = self.update_step(
                    self.params, self.opt_state, batch
                )
            print(f"Initial phase epoch {epoch + 1}, loss: {loss_val:.4f}")

    def standard_training(self, n_epochs=100, patience=5):
        """
        Standard training phase with a training/validation split.
        The validation set is taken from the last n^(1/3) timestamps.
        """
        # Determine the number of unique timestamps.
        T = self.X.shape[0] // 2
        n_val_timestamps = int(np.floor(T ** (1 / 3)))
        n_train_timestamps = T - n_val_timestamps
        d = self.X.shape[1]
        m = self.y.shape[1]
        # Reshape data by timestamps.
        X_grouped = self.X.reshape(T, 2, d)
        y_grouped = self.y.reshape(T, 2, m)
        weights_grouped = self.weights.reshape(T, 2)
        # Flatten training data.
        X_train = X_grouped[:n_train_timestamps].reshape(n_train_timestamps * 2, d)
        y_train = y_grouped[:n_train_timestamps].reshape(n_train_timestamps * 2, m)
        w_train = weights_grouped[:n_train_timestamps].reshape(n_train_timestamps * 2)
        # Flatten validation data.
        X_val = X_grouped[n_train_timestamps:].reshape(n_val_timestamps * 2, d)
        y_val = y_grouped[n_train_timestamps:].reshape(n_val_timestamps * 2, m)
        w_val = weights_grouped[n_train_timestamps:].reshape(n_val_timestamps * 2)

        best_val_rmse = float("inf")
        patience_counter = 0
        for epoch in range(n_epochs):
            for batch in self.get_batches(X_train, y_train, w_train, self.batch_size):
                self.params, self.opt_state, _ = self.update_step(
                    self.params, self.opt_state, batch
                )
            current_rmse = self.rmse(self.params, X_val, y_val, w_val)
            print(f"Epoch {epoch + 1}, Validation RMSE: {current_rmse:.4f}")
            if current_rmse < best_val_rmse:
                best_val_rmse = current_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
