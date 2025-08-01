import os
import warnings
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

class modelFitting:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def checkStationarity(self, series: pd.Series) -> None:
        """
        Check if the given time series is stationary.
        """
        result = adfuller(series)
        pvalue = result[1]
        if pvalue < 0.05:
            print(f"✅ Time series is stationary (p={pvalue:.4e})")
        else:
            print(f"❌ Time series is NOT stationary (p={pvalue:.4e})")
            return
    
    def extractSeries(self) -> pd.Series:
        """
        Extract the 'LogReturn' series for the initialized ticker.
        """
        file_path = os.path.join("data", f"{self.ticker}_data.csv")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file for {self.ticker} not found at {file_path}")

        df = pd.read_csv(file_path)
        if "LogReturn" not in df.columns:
            raise ValueError(f"'LogReturn' column not found in {self.ticker} data")

        return df["LogReturn"].dropna()
    
    def fitARIMA(self, series: pd.Series, auto=True, order=None) -> None:
        """
        Fit an ARIMA model to the given time series.
        """
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation")
        warnings.filterwarnings("ignore", category=UserWarning)
        if auto:
            auto_model = pm.auto_arima(
                series, seasonal=False,
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True
            )
            order = auto_model.order

        if order is None:
            raise ValueError("Either auto=True or an `order` tuple must be provided")

        model = ARIMA(series, order=order).fit()

        # Check fitted ARIMA residuals for stationarity
        resid = model.resid.dropna()

        # 4. stationarity test on residuals
        adf_pval = adfuller(resid)[1]
        if adf_pval >= 0.05:
                print(f"❌ ARIMA residuals may not be stationary (ADF p={adf_pval:.3e})")
        else:
            print(f"✅ ARIMA residuals are stationary (ADF p={adf_pval:.3e})")

        # 5. ARCH effect test
        arch_pval = het_arch(resid)[1]
        if arch_pval >= 0.05:
            print(f"❌ No strong ARCH effect detected in residuals (p={arch_pval:.3e}) (Not worth fitting GARCH model, EWMA Volatility sufficient)")
            fit_garch = False
        else:
            print(f"✅ ARCH effect detected in residuals of (p={arch_pval:.3e}) (Worth fitting GARCH model)")
            fit_garch = True

        diagnostics = {
            "arima_order": order,
            "arima_aic": model.aic,
            "resid_adf_pvalue": adf_pval,
            "resid_arch_pvalue": arch_pval
        }

        return model, diagnostics, fit_garch

    def fitGARCH(self, series: pd.Series, p=1, q=1) -> None:
        """
        Fit a GARCH model to the given time series.
        """
        model = arch_model(series, vol='Garch', p=p, q=q).fit(disp="off")
        self.p = p
        self.q = q
        print(model.summary())
        return model

    def fitEWMA(self, returns: pd.Series, λ: float = 0.94, trading_days: int = 252) -> pd.Series:
        """
        Compute daily EWMA volatility from a returns series.
        
        Args:
          returns: pd.Series of (log-)returns
          λ:       decay factor, commonly 0.94 for daily data
          trading_days: number of trading days per year, typically 252 (for annualization)

        Returns:
          vol_ewma: pd.Series of annualized volatility
        """
        # Compute EWMA of squared returns
        var_ewma = returns.pow(2).ewm(alpha=1-λ, adjust=False).mean()

        # Convert variance → daily vol
        vol_daily = np.sqrt(var_ewma)

        # Annualize
        vol_annual = vol_daily * np.sqrt(trading_days)

        # Report the latest point
        latest = vol_annual.iloc[-1]
        print(f"🔶 Latest EWMA volatility (annualized): {latest:.2%}")
        
        return vol_annual
    
    def fitModel(self, series: pd.Series) -> None:
        """
        Fit the ARIMA (Returns) and either GARCH or EWMA model (Volatility) to the time series data of the initialized ticker.
        """
        arima_model, diagnostics, fit_garch = self.fitARIMA(series, auto=True)

        if fit_garch:
            volatility_model = self.fitGARCH(series)
            print(f"Fitted GARCH model for {self.ticker} with order ({self.p}, {self.q})")
        else:
            print(f"Using ARIMA model for {self.ticker} with EWMA Volatility")
            volatility_model = self.fitEWMA(series)
            print(f"Fitted EWMA model for {self.ticker}")


        return arima_model, volatility_model



if __name__ == "__main__":
    model = modelFitting(ticker="AAPL")
    aaplSeries = model.extractSeries()
    model.checkStationarity(aaplSeries)
    arimaModel, diagnostics, fit_garch = model.fitARIMA(aaplSeries, auto=True)
    