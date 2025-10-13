from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

# Load models function
def load_models(crop_name):
    try:
        return {
            'rf': joblib.load(f'rf_model_{crop_name.lower()}.pkl'),
            'xgb': joblib.load(f'xgb_model_{crop_name.lower()}.pkl'),
            'meta': joblib.load(f'meta_model_{crop_name.lower()}.pkl'),
            'poly': joblib.load(f'poly_{crop_name.lower()}.pkl'),
            'scaler': joblib.load(f'scaler_{crop_name.lower()}.pkl'),
            'features': joblib.load(f'features_{crop_name.lower()}.pkl')
        }
    except Exception as e:
        raise Exception(f"Error loading {crop_name} models: {str(e)}")

# Load models at startup
try:
    groundnut_models = load_models('Groundnut')
    paddy_models = load_models('Paddy')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug raw input
        data = request.get_json()
        print("\n=== RAW INPUT RECEIVED ===")
        print(data)

        # Validate input
        required_fields = ['year', 'crop_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400
        
        crop_name = data['crop_name'].strip().capitalize()
        if crop_name not in ['Groundnut', 'Paddy']:
            return jsonify({'error': 'Invalid crop_name. Use "Groundnut" or "Paddy".'}), 400
        
        # Get models
        models = groundnut_models if crop_name == 'Groundnut' else paddy_models
        
        # Prepare input - ensure all expected features exist
        input_data = {
            'year': int(data['year']),
            'month': data.get('month', 'Unknown'),
            'season': data.get('season', 'Unknown'),
            'crop_type': crop_name,
            'is_groundnut': 1 if crop_name == 'Groundnut' else 0,
            'is_paddy': 1 if crop_name == 'Paddy' else 0
        }

        # Add missing features with defaults
        for feature in models['features']:
            if feature not in input_data:
                if feature in ['avg_temp_deg_c', 'avg_rainfall_mm']:
                    input_data[feature] = 25.0 if 'temp' in feature else 100.0  # Reasonable defaults
                else:
                    input_data[feature] = 0

        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Debug processed features
        print("\n=== PROCESSED DATAFRAME ===")
        print(input_df[models['features']])
        print("Columns:", list(input_df.columns))

        # Transform features
        X_poly = models['poly'].transform(input_df[models['features']])
        X_scaled = models['scaler'].transform(X_poly)
        
        # Debug feature transformation
        print("\n=== TRANSFORMED FEATURES ===")
        print("First 5 values:", X_scaled[0][:5])

        # Get predictions
        rf_pred = models['rf'].predict(X_scaled)
        xgb_pred = models['xgb'].predict(X_scaled)
        final_pred = models['meta'].predict(np.column_stack((rf_pred, xgb_pred)))
        
        # Debug predictions
        print("\n=== PREDICTIONS ===")
        print(f"RF: {rf_pred[0]:.2f}, XGB: {xgb_pred[0]:.2f}, Final: {final_pred[0]:.2f}")

        # Format output (removed multipliers for now)
        price_per_quintal = round(float(final_pred[0]), 2)
        price_per_kg = round(price_per_quintal / 100, 2)
        
        return jsonify({
            'crop_name': crop_name,
            'price_per_kg': price_per_kg,
            'price_per_quintal': price_per_quintal,
            'year': data['year'],
            'month': data.get('month'),
            'season': data.get('season'),
            'model_used': 'Ensemble'
        })

    except Exception as e:
        print(f"\n!!! ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)