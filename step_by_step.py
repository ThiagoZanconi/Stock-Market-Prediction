import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Símbolo de la acción (ej. Apple)
ticker_options = ["ZS=F","^NDX", "CL=F", "GC=F", "AAPL"] #soja, nasdaq_100, oil, gold, apple

stock_dataframe = pd.DataFrame()

for ticker in ticker_options:
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01", interval="1mo")
    if (ticker == ticker_options[0]):
        stock_dataframe.index = data.index
    data[ticker+'_Average_Price'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    stock_dataframe[ticker+'_Average_Price'] = data[ticker+'_Average_Price']
# Mostrar las primeras filas
print(stock_dataframe.head())

n_input = 24   # meses de entrada
n_output = 12  # meses a predecir
target_col = 'AAPL_Average_Price'

df_delta = stock_dataframe.pct_change().dropna()  # o df.diff() si preferís diferencias absolutas

X, y = [], []
for i in range(len(df_delta) - n_input - n_output + 1):
    X.append(df_delta.iloc[i:i+n_input].values)  # (n_input, n_features)
    y.append(df_delta.iloc[i+n_input:i+n_input+n_output][target_col].values)  # (n_output,)

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_input, X.shape[2])))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

last_input = df_delta.iloc[-n_input:].values.reshape(1, n_input, X.shape[2])
delta_pred = model.predict(last_input).flatten()  # predicción de cambios relativos

delta_pred = [0.02, 0.01, -0.01, 0.03, ...]  # 12 meses

precio_actual = stock_dataframe[target_col].iloc[-1]
predicted_prices = []

for delta in delta_pred:
    precio_actual *= (1 + delta)
    predicted_prices.append(precio_actual)


# Graficar
plt.plot(stock_dataframe.index, stock_dataframe[target_col], label='Histórico')
plt.plot(predicted_df.index, predicted_df[f'Predicted_{target_col}'], label='Predicción', linestyle='--')
plt.title(f'Predicción de {target_col} (12 meses a futuro)')
plt.xlabel("Fecha")
plt.ylabel("Precio Promedio")
plt.legend()
plt.grid(True)
plt.show()