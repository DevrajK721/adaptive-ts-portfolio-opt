import os
from typing import Optional
import numpy as np
import pandas as pd

from modelFitting import modelFitting

class ModelDrivenStrategy:
    """Baseline trading strategy using ARIMA return forecasts and
    volatility estimates."""

    def __init__(
        self,
        ticker: str,
        target_vol: float = 0.1,
        leverage_cap: float = 1.0,
        smoothing: float = 0.2,
        refit_interval: int = 20,
    ):
        
        self.ticker = ticker
        self.target_vol = target_vol
        self.leverage_cap = leverage_cap
        self.smoothing = smoothing
        self.refit_interval = refit_interval
        self.steps_since_fit = refit_interval
        self.prev_weight = 0.0
        self.df: Optional[pd.DataFrame] = None
        self.arima_model = None
        self.vol_model = None

    def load_data(self) -> pd.DataFrame:
        """Load precomputed indicator data for the ticker."""
        path = os.path.join("data", f"{self.ticker}_data.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file for {self.ticker} not found at {path}")
        self.df = pd.read_csv(path)
        return self.df

    def fit_models(self):
        """Fit ARIMA for returns and GARCH/EWMA for volatility."""
        mf = modelFitting(self.ticker)
        series = mf.extractSeries()
        self.arima_model, self.vol_model = mf.fitModel(series)
        self.series = series
        self.steps_since_fit = 0
        return self.arima_model, self.vol_model

    def _next_return_forecast(self) -> float:
        """One-step-ahead return forecast."""
        return float(self.arima_model.forecast(steps=1).iloc[0])

    def _next_vol_forecast(self) -> float:
        """Forecast next-day volatility from the fitted model."""
        if hasattr(self.vol_model, "forecast"):
            f = self.vol_model.forecast(horizon=1)
            sigma = np.sqrt(f.variance.iloc[-1, 0])
        else:
            # EWMA volatility -> last value
            sigma = float(self.vol_model.iloc[-1])
        return sigma

    def compute_weight(self, tech_filter: bool = True) -> float:
        """Compute the position weight for the next period."""
        if self.df is None:
            self.load_data()
        if (
            self.arima_model is None
            or self.vol_model is None
            or self.steps_since_fit >= self.refit_interval
        ):
            self.fit_models()
        else:
            self.steps_since_fit += 1

        r_hat = self._next_return_forecast()
        sigma_hat = self._next_vol_forecast()

        raw_weight = r_hat / (sigma_hat ** 2)

        if tech_filter:
            last = self.df.iloc[-1]
            tech_agree = (
                (last.get("norm_RSI_14", 0) > 0).astype(int)
                + (last.get("norm_dist_sma20", 0) > 0).astype(int)
                + (last.get("norm_MACD", 0) > 0).astype(int)
            )
            mask = tech_agree >= 2
        else:
            mask = 1

        weight = raw_weight * mask
        weight *= self.target_vol / sigma_hat
        weight = np.clip(weight, -self.leverage_cap, self.leverage_cap)
        weight = self.smoothing * weight + (1 - self.smoothing) * self.prev_weight
        self.prev_weight = weight
        return weight


class SupervisedSignal:
    """Placeholder for a machine-learning based signal combiner."""

    def __init__(self, model):
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)