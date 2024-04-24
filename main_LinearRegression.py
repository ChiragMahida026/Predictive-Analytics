import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Ask the user for the stock symbol
stock_symbol = input("Please enter the stock symbol: ")

# Append .NS to search in the Indian market
stock_symbol += ".NS"

# Download the data
stock = yf.Ticker(stock_symbol)

# Get historical data
data = stock.history(period="1y")

# Prepare the data for prediction
data['Prediction'] = data['Open'].shift(-1)
data.dropna(inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Prediction'],axis=1), data['Prediction'], test_size=0.9)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the next day's opening price
X_forecast = data.drop(['Prediction'],axis=1)[-1:]
forecast_prediction = model.predict(X_forecast)

print("The predicted opening price for the next trading day is: ", forecast_prediction)

# If the predicted opening price is higher than the last closing price, suggest to buy
# Otherwise, suggest to sell
if forecast_prediction[0] > data['Close'].iloc[-1]:
    print("The predicted opening price is higher than the last closing price. You might want to consider buying.")
else:
    print("The predicted opening price is lower than the last closing price. You might want to consider selling.")
