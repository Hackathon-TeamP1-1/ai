from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import requests
def get_forecast_weather(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    next_day_forecast = data['list'][7]  
    return {
        "T2M": next_day_forecast['main']['temp'],
        "WS2M": next_day_forecast['wind']['speed'],
        "RH2M": next_day_forecast['main']['humidity'],
        "ALLSKY_SFC_SW_DWN": 0.5,  
        "PRECTOTCORR": next_day_forecast['pop'],
        "ALLSKY_KT": 0.6  
    }
model = load_model('Energy_Forcasting.h5')
forecast_weather = get_forecast_weather(31.5, 34.5, "436200292fc50b6345c6ff1649378eb3")

future_input_df = pd.DataFrame([{
    "LAT": 31.5,
    "LON": 34.5,
    "ALLSKY_SFC_SW_DWN": forecast_weather['ALLSKY_SFC_SW_DWN'],
    "WS2M": forecast_weather['WS2M'],
    "T2M": forecast_weather['T2M'],
    "RH2M": forecast_weather['RH2M'],
    "PRECTOTCORR": forecast_weather['PRECTOTCORR'],
    "ALLSKY_KT": forecast_weather['ALLSKY_KT']
}])

future_input_scaled = scaler.transform(future_input_df)

future_input_scaled = np.expand_dims(future_input_scaled, axis=0)

prediction = model.predict(future_input_scaled)
print("Prediction (E_produced) for next day:", prediction[0][0])
