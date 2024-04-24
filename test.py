import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Function to get stock data
def get_stock_data(stock_symbol):
    try:
        # Append .NS to search in the Indian market
        stock_symbol += ".NS"

        # Download the data
        stock = yf.Ticker(stock_symbol)

        # Get historical data
        data = stock.history(period="5y")

        # Prepare the data for prediction
        data = data['Close'].values.reshape(-1, 1)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to create training and testing data
def create_dataset(data, time_step=60, train_size=0.8):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Create the training data set
    train_data = scaled_data[0:int(len(scaled_data) * train_size), :]

    x_train, y_train = [], []
    for i in range(time_step, len(train_data)):
        x_train.append(train_data[i-time_step:i, 0])
        y_train.append(train_data[i, 0])

    # Convert to numpy arrays and reshape
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    test_data = scaled_data[int(len(scaled_data) * train_size) - time_step:, :]
    x_test, y_test = [], data[int(len(scaled_data) * train_size):, :]

    for i in range(time_step, len(test_data)):
        x_test.append(test_data[i-time_step:i, 0])

    # Convert to numpy arrays and reshape
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

# Function to create and train LSTM model
def create_and_train_model(x_train, y_train):
    try:
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=10)
        history = model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)
        print(history.history['loss'])
        return model
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None

# Main function to run the script
def main():
    # Ask the user for the stock symbol
    stock_symbol = input("Please enter the stock symbol: ")

    data = get_stock_data(stock_symbol)
    if data is not None:
        x_train, y_train, x_test, y_test, scaler = create_dataset(data)

        model = create_and_train_model(x_train, y_train)

        # Get the predicted values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE: {rmse}")

        # Get the last 60 days of the closing prices from the training data
        last_60_days = x_train[-1]  # This will have shape (60, 1)

        # Add an extra dimension for the batch size
        last_60_days = np.expand_dims(last_60_days, axis=0)  # This will have shape (1, 60, 1)

        # Get the predicted price
        predicted_price = model.predict(last_60_days)

        # Undo the scaling
        predicted_price = scaler.inverse_transform(predicted_price)

        print(f"The predicted opening price for the next trading day is: {predicted_price[0][0]}")

if __name__ == "__main__":
    main()
