import yfinance as yf
import numpy as np
import pandas as pd
import datetime

def lppl_model(stock_symbol, start_date, end_date):

  data = yf.download(stock_symbol, start=start_date, end=end_date)

  returns = data['Close'].diff() / data['Close'].shift(1)

  model = np.polyfit(np.arange(len(returns)), returns, 3)

  predictions = model[0] * np.exp(model[1] * np.arange(len(returns) + 30)) / (1 + np.exp(model[2] * (np.arange(len(returns) + 30) - model[3])))

  df = pd.DataFrame(data={'date': np.arange(len(predictions)), 'crash': predictions < 0})

  return df

def main():

  stock_symbol = input('Enter the stock symbol: ')

  start_date = input('Enter the start date (YYYY-MM-DD): ')
  end_date = datetime.datetime.now().strftime('%Y-%m-%d')

  predictions = lppl_model(stock_symbol, start_date, end_date)

  print(predictions)

if __name__ == '__main__':
  main() 