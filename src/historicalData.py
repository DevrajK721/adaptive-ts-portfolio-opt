import json
import os 
from typing import Dict
import numpy as np 
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

class historicalData:
    def __init__(self, json_path: str = "secrets/secrets.json"):
        """
        Initializes the historicalData class by loading API credentials and configuration from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing API credentials and configuration.
        """
        with open(json_path) as f:
            creds = json.load(f)
            self.apiKey = creds['alpacaApiKey']
            self.apiSecret = creds['alpacaSecretKey']
            self.baseURL = 'https://paper-api.alpaca.markets'
            self.api = REST(self.apiKey, self.apiSecret, self.baseURL, api_version='v2')
            self.tickers = creds["tickers"]
            self.fromDate = creds["dateFrom"]
            self.toDate = creds["dateTo"]

    def loadData(self):
        """
        Loads historical data into dataframe for the specified tickers from Alpaca API and stores it in a dictionary.
        """
        self.data = {}
        for ticker in self.tickers:
            df = self.api.get_bars(
                ticker,
                TimeFrame.Day,
                start=self.fromDate,
                end=self.toDate,
                adjustment="raw"
            ).df

            # convert index to dates only
            df.index = pd.to_datetime(df.index).date
            df.index.name = "Date"

            # log daily returns using closing price
            df["LogReturn"] = np.log(df["close"] / df["close"].shift(1))

            # Moving averages of close
            df["SMA_20"] = df["close"].rolling(window=20).mean()
            df["SMA_50"] = df["close"].rolling(window=50).mean()
            df["SMA_100"] = df["close"].rolling(window=100).mean()
            df["EMA_8"] = df["close"].ewm(span=8, adjust=False).mean()
            df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()

            # RSI (14)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI_14"] = 100 - (100 / (1 + rs))

            # MACD (12, 26, 9)
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

            # Stochastic Oscillator
            low14 = df["low"].rolling(window=14).min()
            high14 = df["high"].rolling(window=14).max()
            df["Stoch_%K"] = 100 * (df["close"] - low14) / (high14 - low14)
            df["Stoch_%D"] = df["Stoch_%K"].rolling(window=3).mean()

            # ATR
            high_low = df["high"] - df["low"]
            high_close_prev = (df["high"] - df["close"].shift()).abs()
            low_close_prev = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df["ATR_14"] = tr.rolling(window=14).mean()

            # OBV
            direction = np.sign(df["close"].diff()).fillna(0)
            df["OBV"] = (direction * df["volume"]).cumsum()

            # ==============================================================
            # Normalized technical indicators
            # ==============================================================

            # Relative distance to moving averages
            df["norm_dist_sma20"] = df["close"] / df["SMA_20"] - 1
            df["norm_dist_sma50"] = df["close"] / df["SMA_50"] - 1
            df["norm_dist_sma100"] = df["close"] / df["SMA_100"] - 1
            df["norm_dist_ema8"] = df["close"] / df["EMA_8"] - 1
            df["norm_dist_ema21"] = df["close"] / df["EMA_21"] - 1

            # Scale bounded oscillators to [-1, 1]
            df["norm_RSI_14"] = (df["RSI_14"] - 50) / 50
            df["norm_Stoch_%K"] = (df["Stoch_%K"] - 50) / 50
            df["norm_Stoch_%D"] = (df["Stoch_%D"] - 50) / 50

            # Rolling z-score for MACD and MACD signal
            roll = 60
            df["macd_mean"] = df["MACD"].rolling(roll).mean().shift(1)
            df["macd_std"] = df["MACD"].rolling(roll).std().shift(1)
            df["norm_MACD"] = (df["MACD"] - df["macd_mean"]) / df["macd_std"]

            df["macd_signal_mean"] = df["MACD_signal"].rolling(roll).mean().shift(1)
            df["macd_signal_std"] = df["MACD_signal"].rolling(roll).std().shift(1)
            df["norm_MACD_signal"] = (df["MACD_signal"] - df["macd_signal_mean"]) / df["macd_signal_std"]

            # ATR percentage of price
            df["norm_ATR_14"] = df["ATR_14"] / df["close"]

            # OBV rolling z-score of daily change
            df["OBV_diff"] = df["OBV"].diff()
            df["obv_diff_mean"] = df["OBV_diff"].rolling(roll).mean().shift(1)
            df["obv_diff_std"] = df["OBV_diff"].rolling(roll).std().shift(1)
            df["norm_OBV"] = (df["OBV_diff"] - df["obv_diff_mean"]) / df["obv_diff_std"]

            # Volume log transform and rolling z-score
            df["log_volume"] = np.log1p(df["volume"])
            df["log_vol_mean"] = df["log_volume"].rolling(roll).mean().shift(1)
            df["log_vol_std"] = df["log_volume"].rolling(roll).std().shift(1)
            df["norm_volume"] = (df["log_volume"] - df["log_vol_mean"]) / df["log_vol_std"]

            # Clip extreme values to reduce outlier impact
            clip_cols = [
                "norm_RSI_14",
                "norm_Stoch_%K",
                "norm_Stoch_%D",
                "norm_MACD",
                "norm_MACD_signal",
                "norm_ATR_14",
                "norm_OBV",
                "norm_volume",
            ]
            df[clip_cols] = df[clip_cols].clip(-3, 3)

            # prepare final columns
            df.reset_index(inplace=True)
            df.rename(columns={"close": "Close", "volume": "Volume"}, inplace=True)
            df = df[
                [
                    "Date",
                    "Close",
                    "Volume",
                    "LogReturn",
                    "SMA_20",
                    "SMA_50",
                    "SMA_100",
                    "EMA_8",
                    "EMA_21",
                    "RSI_14",
                    "MACD",
                    "MACD_signal",
                    "Stoch_%K",
                    "Stoch_%D",
                    "ATR_14",
                    "OBV",
                    # Normalized features
                    "norm_dist_sma20",
                    "norm_dist_sma50",
                    "norm_dist_sma100",
                    "norm_dist_ema8",
                    "norm_dist_ema21",
                    "norm_RSI_14",
                    "norm_Stoch_%K",
                    "norm_Stoch_%D",
                    "norm_MACD",
                    "norm_MACD_signal",
                    "norm_ATR_14",
                    "norm_OBV",
                    "norm_volume",
                ]
            ]

            # Drop rows with NaN values
            df.dropna(inplace=True)
            self.data[ticker] = df


    def returnCSV(self):
        """
        Outputs the historical data to CSV files for each ticker.
        """
        for ticker, df in self.data.items():
            output_path = f"data/{ticker}_data.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} rows of {ticker} data to {output_path}")

# Example Usage 
if __name__ == "__main__":
    historicalData = historicalData()
    data = historicalData.loadData()
    historicalData.returnCSV()



