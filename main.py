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
target_col = 'GC=F_Average_Price'

contemporary_data = yf.download("GC=F", start="2023-01-01", end="2025-01-01", interval="1mo")
contemporary_data[target_col] = (contemporary_data['Open'] + contemporary_data['High'] + contemporary_data['Low'] + contemporary_data['Close']) / 4
# Normalizamos todos los valores (multivariado)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_dataframe)
scaled_df = pd.DataFrame(scaled_data, index=stock_dataframe.index, columns=stock_dataframe.columns)

# Generamos ventanas de entrenamiento
X, y = [], []
for i in range(len(scaled_df) - n_input - n_output + 1):
    X_window = scaled_df.iloc[i:i + n_input].values  # todas las columnas
    y_window = scaled_df.iloc[i + n_input:i + n_input + n_output][target_col].values
    X.append(X_window)
    y.append(y_window)

X = np.array(X)                   # shape: (n_samples, n_input, n_features)
y = np.array(y)                   # shape: (n_samples, n_output)

# Red LSTM multivariada → salida univariada con 12 pasos
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_input, X.shape[2])))
model.add(Dense(n_output))  # salida con 12 valores del target
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Predicción usando la última ventana de entrada
last_window = scaled_df.iloc[-n_input:].values.reshape(1, n_input, X.shape[2])
predicted_scaled = model.predict(last_window)

# Invertir normalización solo de la columna target
# Creamos un array "falso" del mismo shape que los datos originales para inversa
fake_input = np.zeros((n_output, len(stock_dataframe.columns)))
target_idx = stock_dataframe.columns.get_loc(target_col)
fake_input[:, target_idx] = predicted_scaled.flatten()
predicted_unscaled = scaler.inverse_transform(fake_input)[:, target_idx]

# Armar DataFrame con las predicciones
future_dates = pd.date_range(start=stock_dataframe.index[-1] + pd.offsets.MonthBegin(1), periods=n_output, freq='MS')
predicted_df = pd.DataFrame({f'Predicted_{target_col}': predicted_unscaled}, index=future_dates)

# Graficar
plt.plot(stock_dataframe.index, stock_dataframe[target_col], label='Histórico')
plt.plot(predicted_df.index, predicted_df[f'Predicted_{target_col}'], label='Predicción', linestyle='--')
plt.plot(contemporary_data.index, contemporary_data[target_col], label='Contemporaneo', linestyle=':')
plt.title(f'Predicción de {target_col} (12 meses a futuro)')
plt.xlabel("Fecha")
plt.ylabel("Precio Promedio")
plt.legend()
plt.grid(True)
plt.show()