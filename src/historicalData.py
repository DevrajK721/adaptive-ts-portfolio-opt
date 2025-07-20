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



