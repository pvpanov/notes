import numpy as np
import pandas as pd


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
    n = len(time_index)

    z1 = pd.Series(np.random.randn(n), index=time_index)
    z2 = pd.Series(np.random.randn(n), index=time_index)
    z3 = pd.Series(np.random.normal(0, 0.5, n), index=time_index)  # lighter variance
    z4 = pd.Series(np.random.standard_t(df=3, size=n), index=time_index)  # heavy-tailed

    data = pd.DataFrame({"z1": z1, "z2": z2, "z3": z3, "z4": z4})

    # Compute technical indicators for each base series.
    for i, col in enumerate(["z1", "z2", "z3", "z4"], start=1):
        series = data[col]
        data[f"x{i}_cumdiff"] = cumulative_diff(series, window=25)
        data[f"x{i}_norm"] = normalized_rolling_distance(series, window=50)
        data[f"x{i}_ewm_std"] = ewm_std(series, span=50)
        data[f"x{i}_rsi"] = rsi(series, window=75)

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
    X = data[features].values  # shape (n, 12)

    # Set fixed weights and biases for the two-layer network.
    # We use a different random state for the network weights.
    rng = np.random.RandomState(123)
    W1 = rng.randn(12, 4)  # First layer weights (input_dim=12, neurons=4)
    b1 = rng.randn(4)  # First layer biases
    W2 = rng.randn(4, 2)  # Second layer weights (4 -> 2)
    b2 = rng.randn(2)  # Second layer biases

    # Compute the first hidden layer using leaky ReLU activation.
    H1 = leaky_relu(np.dot(X, W1) + b1)
    # Compute the second hidden layer.
    H2 = leaky_relu(np.dot(H1, W2) + b2)

    # Simulated network outputs before adding noise.
    y1 = H2[:, 0]
    y2 = H2[:, 1]

    # Add heavy-tailed noise to each target.
    corr_noise = np.random.randn(n)
    target1 = y1 + 0.3 * corr_noise + np.random.standard_t(df=3, size=n) * 0.5
    target2 = y2 - 0.4 * corr_noise + np.random.standard_t(df=3, size=n) * 0.5

    # Add the two targets to the DataFrame.
    data["target1"] = target1
    data["target2"] = target2

    return data
