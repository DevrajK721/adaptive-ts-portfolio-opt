import json
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
                adjustment='raw'
            ).df
            self.data[ticker] = df
        return self.data

    def returnCSV(self):
        """
        Outputs the historical data to CSV files for each ticker.
        """
        for ticker, df in self.data.items():
            output_path = f'data/{ticker}_data.csv'
            df.to_csv(output_path, index=True)
            print(f"✅ Saved {len(df)} rows of {ticker} data to {output_path}")

# Example Usage 
if __name__ == "__main__":
    historicalData = historicalData()
    data = historicalData.loadData()
    historicalData.returnCSV()



