from pycaret.regression import *
import yfinance as yf
import pandas as pd

# Símbolo de la acción (ej. Apple)
ticker_options = ["ZS=F","^NDX", "CL=F", "GC=F", "AAPL", "PTON"] #soja, nasdaq_100, oil, gold, apple

stock_dataframe = pd.DataFrame()

for ticker in ticker_options:
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01", interval="1mo")
    if (ticker == ticker_options[0]):
        stock_dataframe.index = data.index
    data[ticker+'_Average_Price'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    stock_dataframe[ticker+'_Average_Price'] = data[ticker+'_Average_Price']
# Mostrar las primeras filas
print(stock_dataframe.head())

target_asset = ticker_options[1]

target_col = target_asset+'_Average_Price'

contemporary_data = yf.download(target_asset, start="2023-01-01", end="2025-01-01", interval="1mo")
contemporary_data[target_col] = (contemporary_data['Open'] + contemporary_data['High'] + contemporary_data['Low'] + contemporary_data['Close']) / 4


s = setup(stock_dataframe, target=target_col)
best_model = compare_models()

predict_model(best_model)
plot_model(best_model) 
