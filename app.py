from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load your trained model
with open('solar_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Solar Radiation Predictor is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Define features in the same order as training
    features = ['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 'PS',
                'CLRSKY_SFC_SW_DWN', 'ALLSKY_KT', 'Month', 'DayOfYear', 'Weekday']
    
    df = pd.DataFrame([data], columns=features)
    df = df.fillna(df.mean())  # Handle missing values
    
    prediction = model.predict(df)
    return jsonify({'predicted_radiation': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
