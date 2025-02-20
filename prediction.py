import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import requests
from tensorflow.keras.models import load_model
model = load_model('Energy_Forcasting.h5')
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
forecast_weather=get_forecast_weather(31.5, 34.5,"2UUY9L43WMFU6BTQSUEWECQ4B")
future_input = np.array([[31.5, 34.5,(forecast_weather['ALLSKY_SFC_SW_DWN']/1000), (forecast_weather['WS2M']/3.6), forecast_weather['T2M'], forecast_weather['RH2M'], forecast_weather['PRECTOTCORR'], forecast_weather['ALLSKY_KT']]])
future_input_df = pd.DataFrame(future_input, columns=["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"])

future_input_df["E_produced"] = 0

future_input_scaled = scaler.transform(future_input_df)
future_input_scaled = future_input_scaled[:, :-1]  

future_input_scaled = np.expand_dims(future_input_scaled, axis=0)

prediction = model.predict(future_input_scaled)
# 115.74*prediction[0][0] in W/m²
