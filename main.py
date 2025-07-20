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
