import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("data_cleaned.csv")
df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
df.set_index('DATE', inplace=True)

panel_area = 1.7
efficiency = 0.20
sun_hours = 5

df["E_produced"] = df["ALLSKY_SFC_SW_DWN"] * panel_area * efficiency * sun_hours


features = ["LAT", "LON","ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"]
target = ["E_produced"]

scaler = MinMaxScaler()
df[features + target] = scaler.fit_transform(df[features + target])

def create_sequences(data, target, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df[features].values, df[target].values, seq_length=5)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

import joblib
joblib.dump(scaler, 'scaler.save')
model.save('Energy_Forcasting.h5')