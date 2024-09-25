import os
import urllib.request, json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Data source
data_source = 'alphavantage'  # alphavantage or kaggle

# Avoid hardcoding API key, fetch it from environment variables
api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'XN2NAZLJG6H6BMXY')

if data_source == 'alphavantage':
    # ====================== Loading Data from Alpha Vantage ==================================
    ticker = "AAPL"
    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
    # Save data to this file
    file_to_save = f'stock_market_data-{ticker}.csv'

    df = pd.read_csv(f'stock_market_data-{ticker}.csv')  # Load your stock data
    data = df['Close'].values  # Use the 'Close' price for prediction

    # Fetch the stock data from the URL
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())

    # Extract the "Time Series (Daily)" data from the JSON
    time_series = data['Time Series (Daily)']

    # Convert the nested JSON data into a Pandas DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')

    # Convert the index to a datetime object (the index contains the dates)
    df.index = pd.to_datetime(df.index)

    # Rename the columns to more descriptive names
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Convert columns to numeric types (they are initially strings)
    df = df.apply(pd.to_numeric)

    # Use the last 5 years of data (or adjust based on your needs)
    df = df[['Open', 'High', 'Low', 'Close']]  # We only need OHLC for prediction
    df = df.iloc[::-1]  # Reverse the DataFrame to have the latest data at the bottom

    # Continue with your code from here:

    # Convert the OHLC prices to a NumPy array
    data = df.values  # Use OHLC prices

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences of past 60 days to predict the OHLC for the next day
    def create_sequences(data, time_steps):
        x, y = [], []
        for i in range(len(data) - time_steps):
            x.append(data[i:i + time_steps])  # past 'time_steps' days as input
            y.append(data[i + time_steps])  # next day OHLC as target
        return np.array(x), np.array(y)


    time_steps = 60  # Use the last 60 days of past data to predict the next day
    x_data, y_data = create_sequences(scaled_data, time_steps)

    # Split into training and test sets (e.g., 80% training, 20% test)
    split = int(0.8 * len(x_data))
    x_train, y_train = x_data[:split], y_data[:split]
    x_test, y_test = x_data[split:], y_data[split:]

    # Reshape the data to fit LSTM input format (samples, time steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))  # 4 features (OHLC)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Build the LSTM model
    model = tf.keras.Sequential()

    # Add an LSTM layer
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 4)))
    model.add(tf.keras.layers.LSTM(units=50))  # Another LSTM layer
    model.add(tf.keras.layers.Dense(4))  # Output layer to predict 4 values (Open, High, Low, Close)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    test_loss = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}")

    # Predict future OHLC stock prices
    predicted_stock_price = model.predict(x_test)

    # Inverse transform to get the actual prices
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Inverse transform the actual test prices
    real_stock_price = scaler.inverse_transform(y_test)

    # Plot the results to compare
    plt.plot(real_stock_price[:, 3], color='red', label='Real AAPL Close Price')  # Only plot 'Close' prices
    plt.plot(predicted_stock_price[:, 3], color='blue', label='Predicted AAPL Close Price')  # Only plot 'Close' prices
    plt.title('AAPL Stock Price Prediction (Close Price)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Predict the next OHLC values based on the last 60 days in the test set
    last_60_days = scaled_data[-60:]  # Get the last 60 days from your dataset
    last_60_days = last_60_days.reshape(1, -1, 4)  # Reshape to fit the LSTM input

    predicted_next_price = model.predict(last_60_days)
    predicted_next_price = scaler.inverse_transform(predicted_next_price)

    print(f"Predicted Next Day's Open, High, Low, and Close: {predicted_next_price[0]}")
