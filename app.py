from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("Energy_Forecasting.h5")
scaler = joblib.load("scaler.save")

@app.route("/")
def home():
    return "Energy Forecasting API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Extract input values
        input_data = np.array([
            [
                data["LAT"], data["LON"],
                data["ALLSKY_SFC_SW_DWN"], data["WS2M"],
                data["T2M"], data["RH2M"], data["PRECTOTCORR"], data["ALLSKY_KT"]
            ]
        ])

        # Convert to DataFrame and scale input data
        input_scaled = scaler.transform(pd.DataFrame(input_data, columns=[
            "LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"
        ]))

        # Reshape input for LSTM model
        input_scaled = np.expand_dims(input_scaled, axis=0)

        # Predict energy output
        prediction = model.predict(input_scaled)

        return jsonify({"predicted_energy": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
