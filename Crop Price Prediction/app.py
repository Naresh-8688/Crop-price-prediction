import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

MONTH_MAP = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
}


def normalize_columns(columns):
    normalized = []
    for column in columns:
        cleaned = str(column).strip().replace('\ufeff', '').lower()
        cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
        cleaned = cleaned.strip('_')
        normalized.append(cleaned)
    return normalized


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_crop_data(crop_name):
    crop_key = crop_name.lower()
    csv_map = {
        'groundnut': BASE_DIR / 'groundnut4.csv',
        'paddy': BASE_DIR / 'paddy4.csv',
    }
    path = csv_map.get(crop_key)
    if not path or not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    frame.columns = normalize_columns(frame.columns)
    frame['crop_name'] = frame.get('crop_name', crop_name)
    frame['month'] = frame.get('month', '').map(lambda value: str(value).strip())
    frame['season'] = frame.get('season', '').map(lambda value: str(value).strip())
    return frame


def clean_numeric_columns(frame):
    for column in [
        'total_area_cultivated_ha',
        'total_production_tonnes',
        'yield_tonnes_ha',
        'avg_rainfall_mm',
        'avg_temp_deg_c',
        'msp_rupee_quintal',
        'avg_market_price_rupee_quintal',
        'inflation_rate',
        'fuel_prices_rupee_l',
        'export_demand_tonnes',
    ]:
        if column in frame.columns:
            frame[column] = frame[column].apply(_safe_float)
    return frame


def _get_historical_values(crop_name, year=None, month=None, season=None):
    data = clean_numeric_columns(load_crop_data(crop_name))
    if data.empty:
        return pd.Series([2500.0], name='avg_market_price_rupee_quintal')

    subset = data.copy()
    if year is not None:
        subset = subset[subset.get('year', pd.Series(dtype='float64')).astype(str).str.startswith(str(year))]

    if month:
        month_key = month.strip().lower()
        subset = subset[subset.get('month', '').astype(str).str.lower() == month_key]
        if subset.empty:
            subset = data[data.get('month', '').astype(str).str.lower() == month_key.replace(' ', '_')]
    if season and not subset.empty:
        season_key = season.strip().lower()
        subset = subset[subset.get('season', '').astype(str).str.lower() == season_key]

    if subset.empty:
        subset = data
    if 'avg_market_price_rupee_quintal' not in subset.columns:
        return pd.Series([2500.0], name='avg_market_price_rupee_quintal')
    return subset['avg_market_price_rupee_quintal']


def fallback_prediction(crop_name, year, month=None, season=None):
    values = _get_historical_values(crop_name, year=year, month=month, season=season)
    if values.empty:
        return 2500.0

    base_price = float(np.median(values))
    if year is not None:
        year_delta = year - 2023
        base_price *= 1 + (year_delta * 0.025)
    return max(base_price, 500.0)


def build_feature_row(crop_name, year, month, season, payload):
    month_name = month or 'January'
    month_num = MONTH_MAP.get(month_name.lower(), 1)
    crop_is_groundnut = 1 if crop_name == 'Groundnut' else 0
    crop_is_paddy = 1 if crop_name == 'Paddy' else 0

    feature_row = {
        'years_since_2000': year - 2000,
        'year_squared': (year - 2000) ** 2,
        'year_normalized': (year - 2000) / 26.0,
        'month_sin': float(np.sin(2 * np.pi * (month_num - 1) / 12)),
        'month_cos': float(np.cos(2 * np.pi * (month_num - 1) / 12)),
        'is_monsoon': 1 if month_num in [6, 7, 8, 9] else 0,
        'total_area_cultivated_ha': _safe_float(payload.get('total_area_cultivated_ha'), 100.0),
        'avg_rainfall_mm': _safe_float(payload.get('avg_rainfall_mm'), 100.0),
        'avg_temp_deg_c': _safe_float(payload.get('avg_temp_deg_c'), 25.0),
        'yield_per_ha': _safe_float(payload.get('yield_per_ha'), 2.0),
        'temp_rainfall_ratio': _safe_float(payload.get('temp_rainfall_ratio'), 0.25),
        'msp_rupee_quintal': _safe_float(payload.get('msp_rupee_quintal'), 2500.0),
        'inflation_rate_%': _safe_float(payload.get('inflation_rate_%'), 5.5),
        'fuel_prices_rupee_l': _safe_float(payload.get('fuel_prices_rupee_l'), 100.0),
        'real_price': _safe_float(payload.get('real_price'), 3000.0),
        'is_groundnut': crop_is_groundnut,
        'is_paddy': crop_is_paddy,
    }

    if season:
        feature_row['season'] = season.strip()
    return feature_row


def _resolve_model_path(name_prefix):
    models_dir = BASE_DIR / 'models'
    if models_dir.exists():
        matches = sorted(models_dir.glob(f'{name_prefix}*.pkl'))
        if matches:
            return matches[-1]

    root_path = BASE_DIR / f'{name_prefix}.pkl'
    return root_path


def load_models(crop_name):
    crop_name = crop_name.strip().capitalize()
    paths = {
        'rf': _resolve_model_path(f'rf_model_{crop_name.lower()}'),
        'xgb': _resolve_model_path(f'xgb_model_{crop_name.lower()}'),
        'meta': _resolve_model_path(f'meta_model_{crop_name.lower()}'),
        'scaler': _resolve_model_path(f'scaler_{crop_name.lower()}'),
        'features': _resolve_model_path(f'features_{crop_name.lower()}'),
    }
    if not all(path.exists() for path in paths.values()):
        return None

    try:
        import joblib

        return {
            'rf': joblib.load(paths['rf']),
            'xgb': joblib.load(paths['xgb']),
            'meta': joblib.load(paths['meta']),
            'scaler': joblib.load(paths['scaler']),
            'features': joblib.load(paths['features']),
        }
    except Exception:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'supported_crops': ['Groundnut', 'Paddy']})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        if 'year' not in payload or 'crop_name' not in payload:
            return jsonify({'error': 'Missing year or crop_name'}), 400

        crop_name = str(payload['crop_name']).strip().capitalize()
        if crop_name not in ['Groundnut', 'Paddy']:
            return jsonify({'error': 'Invalid crop_name. Use "Groundnut" or "Paddy".'}), 400

        year = int(payload['year'])
        month = str(payload.get('month') or '').strip()
        season = str(payload.get('season') or '').strip()

        models = load_models(crop_name)
        historical_price = fallback_prediction(crop_name, year, month=month, season=season)

        if models is not None:
            try:
                feature_row = build_feature_row(crop_name, year, month, season, payload)
                input_df = pd.DataFrame([feature_row])
                features_needed = models['features']
                ordered_df = input_df.reindex(columns=features_needed, fill_value=0.0)
                X_scaled = models['scaler'].transform(ordered_df)
                rf_pred = models['rf'].predict(X_scaled)[0]
                xgb_pred = models['xgb'].predict(X_scaled)[0]
                final_pred = models['meta'].predict(np.column_stack((np.array([rf_pred]), np.array([xgb_pred]))))[0]
                blended_price = (0.7 * final_pred) + (0.3 * historical_price)
                price_per_quintal = round(float(blended_price), 2)
                model_label = 'Ensemble + historical blend'
            except Exception:
                price_per_quintal = round(float(historical_price), 2)
                model_label = 'Fallback'
        else:
            price_per_quintal = round(float(historical_price), 2)
            model_label = 'Fallback'

        price_per_kg = round(price_per_quintal / 100, 2)
        return jsonify({
            'crop_name': crop_name,
            'price_per_kg': price_per_kg,
            'price_per_quintal': round(price_per_quintal, 2),
            'year': year,
            'month': month if month else None,
            'season': season if season else None,
            'model_used': model_label,
        })

    except Exception as exc:  # pragma: no cover - app-level guard
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)