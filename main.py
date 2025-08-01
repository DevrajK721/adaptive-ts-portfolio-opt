# Basic Imports 
import json
import os 
from typing import Dict
import numpy as np 
import pandas as pd
import warnings
import sys 
from alpaca_trade_api.rest import REST, TimeFrame

# Model Fitting Imports
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Class Imports
sys.path.append("src")
from historicalData import historicalData
from modelFitting import modelFitting 
from backtestStrategy import backtest_strategy
from tradingStrategy import LinearRegressionStrategy

# Historical Data Extraction 
historicalData = historicalData()
historicalData.loadData()
historicalData.returnCSV()

# Model Fitting for each ticker 
for ticker in historicalData.data.keys():
    print(f"{'#' * 20} {ticker} {'#' * 20}")
    model = modelFitting(ticker=ticker)
    series = model.extractSeries()
    model.checkStationarity(series)
    arimaModel, volatilityModel = model.fitModel(series)

    result, metrics = backtest_strategy(
        ticker,
        strategy_cls=LinearRegressionStrategy,
    )
    print(
        f"{ticker} Profit: ${metrics['profit']:.2f} ({metrics['profit_pct']:.2f}%) | "
        f"Max Drawdown: {metrics['max_drawdown']:.2%} | "
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
    )

