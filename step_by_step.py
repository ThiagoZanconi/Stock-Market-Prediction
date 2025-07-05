import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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

n_input = 12   # meses de entrada
n_output = 12  # meses a predecir

#Genera un dataframe con las variaciones de fila a fila (La primer fila se elimina, ya que en esta no hay variacion)
df_delta = stock_dataframe.pct_change().dropna() 

X: list[np.ndarray] = []
y: list[np.ndarray] = []
for i in range(len(df_delta) - n_input - n_output + 1):
    X.append(df_delta.iloc[i:i+n_input].to_numpy())  # (n_input, n_features)
    y.append(df_delta.iloc[i+n_input:i+n_input+n_output][target_col].to_numpy())  # (n_output,)

X = np.array(X)
y = np.array(y)

print("Primer ejemplo de X:")
print(X[0]) 

print("\nPrimer ejemplo de y:")
print(y[0]) 

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_input, X.shape[2])))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

last_input = df_delta.iloc[-n_input:].values.reshape(1, n_input, X.shape[2])
delta_pred = model.predict(last_input).flatten()  # predicción de cambios relativos

precio_actual = stock_dataframe[target_col].iloc[-1]
predicted_prices = []

for delta in delta_pred:
    precio_actual *= (1 + delta)
    predicted_prices.append(precio_actual)

real_prices = stock_dataframe[target_col].copy()

# Última fecha real
last_date = stock_dataframe.index[-1]

# Generamos 12 fechas futuras con frecuencia mensual
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_output, freq='MS')

# Graficar
plt.plot(contemporary_data.index, contemporary_data[target_col], label='Contemporaneo', linestyle=':')
plt.plot(stock_dataframe.index, real_prices, label="Precio Real")
plt.plot(future_dates, predicted_prices, 'r--', label="Predicción LSTM (12 meses)")
plt.title(f'Predicción de {target_col} (12 meses a futuro)')
plt.xlabel("Fecha")
plt.ylabel("Precio Promedio")
plt.legend()
plt.grid(True)
plt.show()