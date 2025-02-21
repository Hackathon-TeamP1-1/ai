import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import requests

# Load model and scaler
model = load_model('Energy_Forecasting.h5')
scaler = joblib.load("scaler.save")

def get_forecast_weather(lat, lon, api_key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/tomorrow?unitGroup=metric&key={api_key}&contentType=json"
    response = requests.get(url)
    data = response.json()
    next_day_forecast = data['days'][0]
    return {
        "T2M": next_day_forecast['temp'],
        "WS2M": next_day_forecast['windspeed'],
        "RH2M": next_day_forecast['humidity'],
        "ALLSKY_SFC_SW_DWN": next_day_forecast['solarradiation'],
        "PRECTOTCORR": next_day_forecast['precip'],
        "ALLSKY_KT": 0.625
    }

forecast_weather = get_forecast_weather(31.5, 34.5, "2UUY9L43WMFU6BTQSUEWECQ4B")
future_input = np.array([[31.5, 34.5, forecast_weather['ALLSKY_SFC_SW_DWN'], forecast_weather['WS2M'], 
                          forecast_weather['T2M'], forecast_weather['RH2M'], forecast_weather['PRECTOTCORR'], forecast_weather['ALLSKY_KT']]])

future_input_df = pd.DataFrame(future_input, columns=["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"])

# Scale input data (ONLY use features, DO NOT add `E_produced`)
future_input_scaled = scaler.transform(future_input_df)

# Reshape for LSTM model
future_input_scaled = np.expand_dims(future_input_scaled, axis=0)

# Predict energy output
prediction = model.predict(future_input_scaled)

print(f"🔮 Predicted Energy Output: {prediction[0][0]} KW/m²")
