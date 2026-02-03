from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')

# Load models function
def load_models(crop_name):
    try:
        return {
            'rf': joblib.load(f'rf_model_{crop_name.lower()}.pkl'),
            'xgb': joblib.load(f'xgb_model_{crop_name.lower()}.pkl'),
            'meta': joblib.load(f'meta_model_{crop_name.lower()}.pkl'),
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
        
        # Prepare input with feature engineering
        year = int(data['year'])
        month_str = data.get('month', 'January')
        
        # Map month name to number
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_num = month_map.get(month_str, 1)
        
        # Feature engineering matching training pipeline
        input_data = {
            'years_since_2000': year - 2000,
            'year_squared': (year - 2000) ** 2,
            'year_normalized': (year - 2000) / 26.0,  # Approximate normalization
            'month_sin': np.sin(2 * np.pi * (month_num - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (month_num - 1) / 12),
            'is_monsoon': 1 if month_num in [6, 7, 8, 9] else 0,
            'total_area_cultivated_ha': float(data.get('total_area_cultivated_ha', 100.0)),
            'avg_rainfall_mm': float(data.get('avg_rainfall_mm', 100.0)),
            'avg_temp_deg_c': float(data.get('avg_temp_deg_c', 25.0)),
            'yield_per_ha': float(data.get('yield_per_ha', 2.0)),
            'temp_rainfall_ratio': float(data.get('temp_rainfall_ratio', 0.25)),
            'msp_rupee_quintal': float(data.get('msp_rupee_quintal', 2500.0)),
            'inflation_rate_%': float(data.get('inflation_rate_%', 5.5)),
            'fuel_prices_rupee_l': float(data.get('fuel_prices_rupee_l', 100.0)),
            'real_price': float(data.get('real_price', 3000.0)),  # Default historical average
            'is_groundnut': 1 if crop_name == 'Groundnut' else 0,
            'is_paddy': 1 if crop_name == 'Paddy' else 0
        }

        # Create DataFrame with only features needed
        input_df = pd.DataFrame([input_data])
        features_needed = models['features']
        
        # Select only features that exist in the loaded models
        input_df_filtered = input_df[[col for col in features_needed if col in input_df.columns]]
        
        # Debug processed features
        print("\n=== PROCESSED DATAFRAME ===")
        print(input_df_filtered)
        print("Features used:", list(input_df_filtered.columns))
        print("Expected features:", features_needed)

        # Scale features (no polynomial transformation needed - already done during training)
        X_scaled = models['scaler'].transform(input_df_filtered[features_needed])
        
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

        # Format output
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)