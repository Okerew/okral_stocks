import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import yfinance as yf
import datetime


print("Welcome to Okral Stock Algorithm")

ticker_symbol = input("Enter the stock symbol: ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

provided_end_date = input("Enter the end date (YYYY-MM-DD): ")
if provided_end_date > end_date:
    end_date = provided_end_date

data = yf.download(ticker_symbol, start=start_date, end=end_date)

data = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

lookback = int(input("Enter how many days you want to look back for prediction: "))

X = []
y = []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

num_future_dates = int(input("Enter the number of future days you want to predict: "))
X_future = scaled_data[-lookback:].reshape(1, -1, 1)

future_predictions = []

for _ in range(num_future_dates):
    prediction = model.predict(X_future)
    scaled_data = np.append(scaled_data, prediction)
    X_future = scaled_data[-lookback:].reshape(1, -1, 1)
    future_prediction = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    future_predictions.append(future_prediction)
    end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
    end_date = end_date.strftime('%Y-%m-%d')

last_date = pd.to_datetime(end_date)
future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=num_future_dates)

plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(range(len(data), len(data) + len(future_predictions)), future_predictions, label='Predicted', linestyle='--')
plt.axvline(x=len(data) - 1, color='green', linestyle='--', linewidth=1, label='Last Available Date')
plt.xticks(rotation=45)
plt.title('Symbol: ' + ticker_symbol)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
