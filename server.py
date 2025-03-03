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
        data = request.json  # Get request data

        # Ensure all required keys exist
        required_keys = ["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input fields", "received_data": data}), 400

        # Prepare input for model
        input_data = np.array([[data[key] for key in required_keys]])

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data, columns=required_keys)

        # 🔴 **Fix:** Add `E_produced` column (placeholder) before scaling
        input_df["E_produced"] = 0  

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # **Remove `E_produced` after scaling**
        input_scaled = input_scaled[:, :-1]  

        # Reshape for LSTM model (Ensure correct shape: [1, sequence_length, features])
        input_scaled = np.expand_dims(input_scaled, axis=0)

        # Predict energy output
        prediction = model.predict(input_scaled)

        return jsonify({"predicted_energy": float(prediction[0][0])})

    except Exception as e:
        print("❌ API Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)