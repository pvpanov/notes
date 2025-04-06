import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_time_index(n_days=10, start_date="2015-01-01"):
    bdays = pd.bdate_range(start=start_date, periods=n_days)
    timestamps = []
    for day in bdays:
        day_str = day.strftime("%Y-%m-%d")
        day_times = pd.date_range(
            start=f"{day_str} 08:45", end=f"{day_str} 15:45", freq="15min"
        )
        timestamps.extend(day_times)
    return pd.DatetimeIndex(timestamps)


def cumulative_diff(series, window=25):
    diff = series.diff(window)
    return diff.cumsum()


def normalized_rolling_distance(series, window=50):
    cum_series = series.cumsum()
    rolling = cum_series.rolling(window, min_periods=window // 2)
    roll_min = rolling.min()
    roll_max = rolling.max()
    norm = np.minimum((cum_series - roll_min) / (roll_max - roll_min), 1)
    return norm


def ewm_std(series, span=50):
    return series.ewm(span=span, adjust=False).std()


def rsi(series, window=75):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window // 2).mean()
    avg_loss = loss.rolling(window, min_periods=window // 2).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def generate_synthetic_timeseries(n_days=10, seed=42):
    np.random.seed(seed)
    time_index = generate_time_index(n_days=n_days)
    groups = ["A", "B"]
    scaler = StandardScaler()
    data_list = []

    for group in groups:
        n = len(time_index)

        z1 = pd.Series(np.random.randn(n), index=time_index)
        z2 = pd.Series(np.random.randn(n), index=time_index)
        z3 = pd.Series(
            np.random.normal(0, 0.5, n), index=time_index
        )  # lighter variance
        z4 = pd.Series(
            np.random.standard_t(df=3, size=n), index=time_index
        )  # heavy-tailed

        df = pd.DataFrame({"z1": z1, "z2": z2, "z3": z3, "z4": z4})

        # Compute technical indicators for each base series.
        for i, col in enumerate(["z1", "z2", "z3", "z4"], start=1):
            series = df[col]
            df[f"x{i}_cumdiff"] = cumulative_diff(series, window=25)
            df[f"x{i}_norm"] = normalized_rolling_distance(series, window=50)
            df[f"x{i}_ewm_std"] = ewm_std(series, span=50)
            df[f"x{i}_rsi"] = rsi(series, window=75)
        df = pd.DataFrame(
            scaler.fit_transform(df.to_numpy()), columns=df.columns, index=df.index
        )

        # Build a non-linear mapping (a two-layer network) to generate two targets.
        # Use only features from z1, z3, and z4 (thus x2 features are omitted).
        features = [
            "x1_cumdiff",
            "x1_norm",
            "x1_ewm_std",
            "x1_rsi",
            "x3_cumdiff",
            "x3_norm",
            "x3_ewm_std",
            "x3_rsi",
            "x4_cumdiff",
            "x4_norm",
            "x4_ewm_std",
            "x4_rsi",
        ]
        X = df[features].values  # shape (n, 12)

        # Set fixed weights and biases for the two-layer network.
        # We use a different random state for the network weights.
        rng = np.random.RandomState(123)
        W0 = rng.randn(12, 2)
        b0 = rng.randn(2)
        W1 = rng.randn(12, 4)  # First layer weights (input_dim=12, neurons=4)
        b1 = rng.randn(4)  # First layer biases
        W2 = rng.randn(4, 2)  # Second layer weights (4 -> 2)
        b2 = rng.randn(2)  # Second layer biases

        # wide part
        H0 = np.dot(X, W0) + b0
        # Compute the first hidden layer using leaky ReLU activation.
        H1 = leaky_relu(np.dot(X, W1) + b1)
        # Compute the second hidden layer.
        H2 = leaky_relu(np.dot(H1, W2) + b2)

        # Simulated network outputs before adding noise.
        y1 = scaler.fit_transform((H0[:, 0] + H2[:, 0])[:, None]).ravel()
        y2 = scaler.fit_transform((H0[:, 1] + H2[:, 1])[:, None]).ravel()

        # Add heavy-tailed noise to each target.
        corr_noise = np.random.randn(n)
        target1 = y1 + corr_noise + np.random.standard_t(df=3, size=n)
        target2 = y2 - corr_noise + np.random.standard_t(df=3, size=n)

        # Add the two targets to the DataFrame.
        df["target1"] = target1
        df["target2"] = target2

        # Add the group label.
        df["underlying"] = group
        data_list.append(df)

    final_data = pd.concat(data_list).sort_index().set_index("underlying", append=True)

    w_a = (1 + np.random.uniform(0, 1)) / 3
    weights = [w_a, 1 - w_a]
    assert len(weights) == len(groups)
    final_data["weight"] = np.tile(weights, len(final_data) // 2)
    return final_data
