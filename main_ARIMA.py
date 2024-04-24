import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Ask the user for the stock symbol
stock_symbol = input("Please enter the stock symbol: ")

# Append .NS to search in the Indian market
stock_symbol += ".NS"

# Download the data
stock = yf.Ticker(stock_symbol)

# Get historical data
data = stock.history(period="1y")

# Prepare the data for prediction
data = data['Close'].values

# Fit the ARIMA model
model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()

# Make prediction
prediction = model_fit.forecast()[0]

print('Predicted Stock Price: %f' % prediction)
