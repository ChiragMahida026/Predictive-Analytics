import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU  # Import GRU instead of LSTM
from keras.layers import Bidirectional , LSTM

# Ask the user for the stock symbol
stock_symbol = input("Please enter the stock symbol: ")

# Append .NS to search in the Indian market
stock_symbol += ".NS"

# Download the data
stock = yf.Ticker(stock_symbol)

# Get historical data
data = stock.history(period="5y")

# Use more features for prediction
data_features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Scale the data
scaler_features = MinMaxScaler(feature_range=(0,1))
scaled_data_features = scaler_features.fit_transform(data_features)

# Prepare the data for prediction
data_close = data['Close'].values.reshape(-1, 1)

# Scale the closing prices
scaler_close = MinMaxScaler(feature_range=(0,1))
scaled_data_close = scaler_close.fit_transform(data_close)

# Create the training data set
train_data = scaled_data_features[0:int(len(scaled_data_features)*0.8), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :])
    y_train.append(scaled_data_close[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# # Build the stacked GRU model
# model = Sequential()
# model.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # Use GRU instead of LSTM
# model.add(GRU(50, return_sequences=False))  # Use GRU instead of LSTM
# model.add(Dense(25))
# model.add(Dense(1))

# Build the Bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=50)

# Test the model
test_data = scaled_data_features[int(len(scaled_data_features)*0.8) - 60:, :]
x_test = []
y_test = data_close[int(len(scaled_data_features)*0.8):, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# Get the predicted values
predictions = model.predict(x_test)
predictions = scaler_close.inverse_transform(predictions)  # Use the scaler for closing prices

# Print the predictions
next_day_opening_price = predictions[-1][0]
print(f"The predicted opening price for the next trading day is: {next_day_opening_price}")
