"""Data loading and LSTM forecasting for household active power."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
SAMPLE_INTERVAL = "15min"
LOOKBACK_STEPS = 96
FORECAST_STEPS = 16


def download_uci_dataset(destination: str | Path) -> Path:
    """Download the original UCI archive if it is not already present."""
    archive = Path(destination)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urlretrieve(UCI_URL, archive)
    return archive


def load_uci_dataset(archive_path: str | Path) -> pd.Series:
    """Read the UCI archive and return complete 15-minute active-power means."""
    with ZipFile(archive_path) as archive:
        member = next(
            name for name in archive.namelist()
            if name.lower().endswith("household_power_consumption.txt")
        )
        with archive.open(member) as source:
            raw = pd.read_csv(source, sep=";", na_values="?", low_memory=False)

    timestamps = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    power = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    readings = pd.Series(power.to_numpy(), index=timestamps, name="power_kw")
    readings = readings[readings.index.notna()].sort_index()
    readings = readings[~readings.index.duplicated(keep="last")]
    readings = readings.resample(SAMPLE_INTERVAL).mean().interpolate(method="time")
    return readings.dropna().clip(lower=0)


def make_demo_dataset(periods: int = 14 * 24 * 4) -> pd.Series:
    """Create a deterministic, clearly synthetic series for the dashboard preview."""
    rng = np.random.default_rng(21)
    index = pd.date_range(end=pd.Timestamp.now().floor("15min"), periods=periods, freq=SAMPLE_INTERVAL)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    daily = 0.48 + 0.34 * np.exp(-((hour - 7.5) / 2.1) ** 2)
    daily += 1.05 * np.exp(-((hour - 19.5) / 2.8) ** 2)
    weekly = 0.12 * (1 + np.sin(2 * np.pi * index.dayofweek.to_numpy() / 7))
    spikes = (rng.random(periods) < 0.025) * rng.uniform(0.5, 2.4, periods)
    noise = rng.normal(0, 0.09, periods)
    values = np.maximum(0.2, daily + weekly + spikes + noise)
    return pd.Series(values, index=index, name="power_kw")


def train_lstm(
    readings: pd.Series,
    epochs: int = 8,
    max_training_windows: int = 10_000,
) -> tuple[object, float, float, list[float]]:
    """Train a direct multi-step LSTM and return model, scale, range, and losses."""
    import tensorflow as tf

    values = readings.to_numpy(dtype=np.float32)
    window_size = LOOKBACK_STEPS + FORECAST_STEPS
    if len(values) < window_size + 32:
        raise ValueError("At least 128 complete 15-minute readings are needed to train.")

    split = int(len(values) * 0.85)
    train_values = values[:split]
    minimum = float(train_values.min())
    value_range = max(float(train_values.max() - minimum), 1e-6)
    scaled = (values - minimum) / value_range
    windows = np.lib.stride_tricks.sliding_window_view(scaled, window_size)
    train_window_count = max(1, split - window_size + 1)
    train_windows = windows[:train_window_count]
    if len(train_windows) > max_training_windows:
        selected = np.linspace(0, len(train_windows) - 1, max_training_windows, dtype=int)
        train_windows = train_windows[selected]
    validation_start = max(0, split - LOOKBACK_STEPS)
    validation_windows = windows[validation_start:]
    if len(validation_windows) == 0:
        raise ValueError("Not enough readings remain for a future validation period.")
    validation_windows = validation_windows[:: max(1, len(validation_windows) // 1500)]

    train_x = train_windows[:, :LOOKBACK_STEPS, np.newaxis]
    train_y = train_windows[:, LOOKBACK_STEPS:]
    validation_x = validation_windows[:, :LOOKBACK_STEPS, np.newaxis]
    validation_y = validation_windows[:, LOOKBACK_STEPS:]

    tf.keras.utils.set_random_seed(21)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK_STEPS, 1)),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(FORECAST_STEPS),
        ]
    )
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=0,
    )
    prediction = model.predict(validation_x, verbose=0)
    mae_kw = float(np.mean(np.abs(prediction - validation_y)) * value_range)
    losses = [float(loss) for loss in history.history["loss"]]
    return model, minimum, value_range, losses + [mae_kw]


def forecast_lstm(
    model: object,
    readings: pd.Series,
    minimum: float,
    value_range: float,
) -> pd.Series:
    """Forecast the next four hours using the most recent 24 hours."""
    recent = readings.tail(LOOKBACK_STEPS).to_numpy(dtype=np.float32)
    if len(recent) < LOOKBACK_STEPS:
        raise ValueError("At least 24 hours of readings are needed for a forecast.")
    scaled = (recent - minimum) / value_range
    prediction = model.predict(scaled.reshape(1, LOOKBACK_STEPS, 1), verbose=0)[0]
    index = pd.date_range(
        start=readings.index[-1] + pd.Timedelta(15, unit="min"),
        periods=FORECAST_STEPS,
        freq=SAMPLE_INTERVAL,
    )
    return pd.Series(np.maximum(prediction * value_range + minimum, 0), index=index, name="power_kw")
