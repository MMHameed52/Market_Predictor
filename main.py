from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  # This code has been tested with TensorFlow 1.6

# Define global variables
data_source = 'alphavantage'  # 'alphavantage' or 'kaggle'
api_key = 'XN2NAZLJG6H6BMXY'
ticker = "AAL"
file_to_save = 'stock_market_data-%s.csv' % ticker
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (ticker, api_key)

# Step 1: Load Data
def load_data():
    if not os.path.exists(file_to_save):
        print("Fetching new data from Alpha Vantage API")
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                            float(v['4. close']), float(v['1. open'])]
                df.loc[-1, :] = data_row
                df.index = df.index + 1

        # Sort and save the data
        df = df.sort_values('Date')
        df.to_csv(file_to_save)
        print(f'Data saved to : {file_to_save}')
    else:
        print("Loading data from existing CSV file")
        df = pd.read_csv(file_to_save)
    return df

# Step 2: Visualize Data
def visualize_data(df):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()

# Step 3: Normalize and Scale Data
def scale_data(df):
    high_prices = df.loc[:, 'High'].values
    low_prices = df.loc[:, 'Low'].values
    mid_prices = (high_prices + low_prices) / 2.0

    # Adjust train-test split based on available data points (80% for train, 20% for test)
    split_ratio = 0.8
    split_point = int(len(mid_prices) * split_ratio)

    # Now split the data
    train_data = mid_prices[:split_point]
    test_data = mid_prices[split_point:]

    # Print the shapes to verify
    print(f"Train data size: {train_data.shape}")
    print(f"Test data size: {test_data.shape}")

    if test_data.size == 0:
        print("Warning: test_data is empty, check your data size.")
        return train_data, None

    # Reshape the data for the scaler
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    # Normalize the training and test data using MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the training data
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    # Transform the test data
    test_data = scaler.transform(test_data)

    # Flatten the data back for further processing
    train_data = train_data.flatten()
    test_data = test_data.flatten()

    return train_data, test_data

# Step 4: Apply Exponential Moving Average Smoothing
def apply_ema_smoothing(train_data):
    EMA = 0.0
    gamma = 0.1
    for ti in range(len(train_data)):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA
    return train_data

# Step 5: Calculate Moving Averages and Predictions
def moving_avg_predictions(train_data, window_size):
    N = train_data.size
    predictions = []
    mse_errors = []

    for pred_idx in range(window_size, N):
        predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
        mse_errors.append((predictions[-1] - train_data[pred_idx]) ** 2)

    mse_error = 0.5 * np.mean(mse_errors)
    print(f'MSE error for standard averaging: {mse_error:.5f}')
    return predictions, mse_error

# Step 6: Visualize the Results
def plot_predictions(df, predictions, window_size, train_data):
    plt.figure(figsize=(18, 9))

    # Plot only the portion of 'df' that corresponds to 'train_data'
    trimmed_dates = df['Date'].iloc[:len(train_data)]

    plt.plot(trimmed_dates, train_data, color='b', label='True Data')
    plt.plot(trimmed_dates[window_size:], predictions, color='orange', label='Predictions')

    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(rotation=45)
    plt.show()

# Data Generator Class for Sequential Data
class DataGeneratorSeq(object):
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))

# Main Execution Flow
if data_source == 'alphavantage':
    df = load_data()  # Load the data
    visualize_data(df)  # Initial visualization
    train_data, test_data = scale_data(df)  # Scaling and normalization
    train_data = apply_ema_smoothing(train_data)  # Apply EMA smoothing

    # Initialize Data Generator for sequential batches
    dg = DataGeneratorSeq(train_data, batch_size=5, num_unroll=5)

    # Unroll batches for further use
    u_data, u_labels = dg.unroll_batches()

    # Debugging: Print unrolled data
    for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
        print(f'\n\nUnrolled index {ui}')
        print('\tInputs: ', dat)
        print('\tOutput:', lbl)

    # Continue with your predictions and plotting, etc.
    predictions, mse_error = moving_avg_predictions(train_data, window_size=100)  # Calculate predictions
    plot_predictions(df, predictions, window_size=100, train_data=train_data)  # Visualize predictions

