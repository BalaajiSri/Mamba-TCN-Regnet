from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradingStep:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class TradingEnvironment:
    DERIVED_FEATURES = {
        "Return",
        "LogReturn",
        "HLRange",
        "OCChange",
        "VolumeChange",
        "RollingVolatility_7",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        window_size: int = 30,
        transaction_cost: float = 0.001,
        initial_cash: float = 1000.0,
        allow_short: bool = True,
        max_steps: int | None = None,
    ):
        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.max_steps = max_steps
        self.position = 0
        self.cash = initial_cash
        self.current_step = 0
        self.steps_taken = 0
        self.equity_curve = []

    def reset(self) -> np.ndarray:
        self.position = 0
        self.cash = self.initial_cash
        self.current_step = self.window_size
        self.steps_taken = 0
        self.equity_curve = [self.initial_cash]
        return self._get_observation()

    def _get_price(self, index: int) -> float:
        return float(self.data.iloc[index]["Close"])

    def _get_observation(self) -> np.ndarray:
        window = self.data.iloc[self.current_step - self.window_size : self.current_step]
        feature_rows = []
        for column in self.feature_columns:
            values = self._feature_values(window, column)
            if column == "Volume":
                values = values / 1e9
            feature_rows.append(values)
        features = np.stack(feature_rows, axis=0).astype(np.float32)
        return features

    def _feature_values(self, window: pd.DataFrame, column: str) -> np.ndarray:
        if column in window.columns:
            return window[column].to_numpy(dtype=np.float32)
        if column not in self.DERIVED_FEATURES:
            raise KeyError(f"Feature '{column}' is neither a data column nor a supported derived feature.")

        close = window["Close"].to_numpy(dtype=np.float32)
        open_price = window["Open"].to_numpy(dtype=np.float32)
        high = window["High"].to_numpy(dtype=np.float32)
        low = window["Low"].to_numpy(dtype=np.float32)
        previous_close = np.roll(close, 1)
        previous_close[0] = close[0]
        close_return = (close - previous_close) / np.maximum(previous_close, 1e-6)

        if column == "Return":
            return close_return
        if column == "LogReturn":
            return np.log(np.maximum(close, 1e-6) / np.maximum(previous_close, 1e-6))
        if column == "HLRange":
            return (high - low) / np.maximum(close, 1e-6)
        if column == "OCChange":
            return (close - open_price) / np.maximum(open_price, 1e-6)
        if column == "RollingVolatility_7":
            values = []
            for idx in range(close_return.shape[0]):
                start = max(0, idx - 6)
                values.append(float(np.std(close_return[start : idx + 1])))
            return np.asarray(values, dtype=np.float32)

        volume = window["Volume"].to_numpy(dtype=np.float32)
        previous_volume = np.roll(volume, 1)
        previous_volume[0] = volume[0]
        return (volume - previous_volume) / np.maximum(previous_volume, 1.0)

    def _resolve_target_position(self, action: int) -> int:
        if action == 2:
            return 1
        if action == 0:
            return -1 if self.allow_short else 0
        return 0

    def step(self, action: int) -> TradingStep:
        current_price = self._get_price(self.current_step - 1)
        next_price = self._get_price(self.current_step)

        old_position = self.position
        self.position = self._resolve_target_position(action)
        trade_penalty = self.transaction_cost * abs(self.position - old_position) * current_price

        reward = self.position * (next_price - current_price) - trade_penalty
        self.cash += reward
        equity = self.cash
        self.equity_curve.append(equity)

        self.current_step += 1
        self.steps_taken += 1

        done = self.current_step >= len(self.data)
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            done = True

        observation = self._get_observation() if not done else np.zeros((len(self.feature_columns), self.window_size), dtype=np.float32)
        info = {
            "price": next_price,
            "position": self.position,
            "cash": self.cash,
            "equity": equity,
        }
        return TradingStep(observation=observation, reward=float(reward), done=done, info=info)

