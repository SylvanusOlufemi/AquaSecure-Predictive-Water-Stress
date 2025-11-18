# app.py - Flask Application for API and UI

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'water_stress_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print("Water Stress Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Run aqua_predict.py first.")
    model = None

# --- 1. Web Route (Simple UI for Demonstration) ---
@app.route('/')
def index():
    return render_template('index.html') # Need a simple HTML file

# --- 2. API Endpoint for Prediction ---
@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        # Get data from POST request (e.g., from C-WIV app or dashboard)
        data = request.get_json(force=True)
        
        # Ensure data structure matches the features the model was trained on
        features = ['Rainfall_Annual_mm', 'Temp_Avg_C', 'Soil_Moisture_Index', 'Population_Density']
        
        # Create input array from request data
        input_data = [data[f] for f in features]
        
        # Reshape for model prediction
        prediction_input = np.array(input_data).reshape(1, -1)
        
        # Get the prediction
        stress_prediction = model.predict(prediction_input)[0]
        
        # Format output
        result = {
            'status': 'success',
            'water_stress_index': round(stress_prediction, 2),
            'risk_level': 'High' if stress_prediction > 60 else ('Medium' if stress_prediction > 30 else 'Low')
        }
        return jsonify(result)
    
    except KeyError as e:
        return jsonify({"error": f"Missing data key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create a simple 'templates' folder and an 'index.html' file to run this
    # Example index.html would contain a form to submit the 4 features
    app.run(debug=True)