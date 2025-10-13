# crop_model.py - Core ML functionality
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging
from typing import Dict, Tuple

class CropPricePredictor:
    """Core machine learning model for crop price prediction"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.models = {}
        self.scalers = {}
        self.features = {}

    def _setup_logger(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def load_data(self, groundnut_path: str, paddy_path: str) -> pd.DataFrame:
        """Load and merge datasets"""
        try:
            self.logger.info("Loading datasets...")
            groundnut = pd.read_csv(groundnut_path)
            paddy = pd.read_csv(paddy_path)
            
            # Standardization and merging
            groundnut = self._clean_data(groundnut)
            paddy = self._clean_data(paddy)
            
            groundnut['crop_type'] = 'Groundnut'
            paddy['crop_type'] = 'Paddy'
            
            return pd.concat([groundnut, paddy], axis=0)
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data with column verification"""
        original_cols = set(df.columns)
    
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('(', '')
            .str.replace(')', '')
            .str.replace('/', '_')
        )
    
    # Verify critical columns
        expected_columns = {
            'avg_market_price_rupee_quintal': [
                'avg_market_price_rupee_quintal',
                'avgmarketprice_rupee_quintal',
                'avgmarketpricerupeequintal'
            ],
            'msp_rupee_quintal': [
                'msp_rupee_quintal',
                'msprupeequintal'
            ]
        }
    
        for target, possible_names in expected_columns.items():
            if not any(name in df.columns for name in possible_names):
                raise ValueError(
                    f"Could not find {target} column after cleaning. "
                    f"Original columns: {original_cols}, "
                    f"Cleaned columns: {list(df.columns)}"
                )
    
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline"""
        # Temporal features
        if 'year' in df.columns:
            df['years_since_2000'] = df['year'] - 2000
            df['year_squared'] = df['years_since_2000']**2
            df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # Month features
        if 'month' in df.columns:
            month_map = {month: idx+1 for idx, month in enumerate([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])}
            df['month_num'] = df['month'].map(month_map).fillna(0)
            df['month_sin'] = np.sin(2 * np.pi * (df['month_num']-1)/12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month_num']-1)/12)
            df['is_monsoon'] = df['month'].isin(['June','July','August','September']).astype(int)
        
        # Agricultural features
        if all(col in df.columns for col in ['total_production_tonnes', 'total_area_cultivated_ha']):
            df['yield_per_ha'] = np.where(
                df['total_area_cultivated_ha'] > 0,
                df['total_production_tonnes'] / df['total_area_cultivated_ha'],
                0
            )
        
        return df

    def train_models(self, df: pd.DataFrame) -> None:
        """Train and save models for all crops"""
        try:
            processed_df = self.preprocess_data(df)
            
            for crop in ['Groundnut', 'Paddy']:
                self.logger.info(f"Training {crop} model...")
                crop_df = processed_df[processed_df['crop_type'] == crop].copy()
                
                X, y, scaler, features = self._prepare_model_data(crop_df)
                rf_model, xgb_model, meta_model = self._train_ensemble(X, y)
                
                # Save artifacts
                self._save_models(crop, rf_model, xgb_model, meta_model, scaler, features)
                self.logger.info(f"{crop} model training complete")
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

def _prepare_model_data(self, df: pd.DataFrame) -> Tuple:
    """Prepare features with enhanced validation"""
    # Find target column by possible names
    target_col = None
    for possible in ['avg_market_price_rupee_quintal', 'avgmarketprice_rupee_quintal']:
        if possible in df.columns:
            target_col = possible
            break
    
    if not target_col:
        raise ValueError(
            "Could not find target price column. "
            f"Available columns: {list(df.columns)}"
        )
    
    numeric_cols = [
        'years_since_2000', 'year_squared', 'year_normalized',
        'month_sin', 'month_cos', 'is_monsoon',
        'total_area_cultivated_ha', 'avg_rainfall_mm', 'avg_temp_deg_c',
        'yield_per_ha', 'temp_rainfall_ratio',
        'msp_rupee_quintal', 'inflation_rate_%', 'fuel_prices_rupee_l'
    ]
    
    # Only use features that exist
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    X = df[numeric_cols]
    y = df[target_col]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, numeric_cols

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Train ensemble of models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
        xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
        
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        
        # Stacking
        stack_features = np.column_stack((rf.predict(X_test), xgb.predict(X_test)))
        meta_model = LinearRegression()
        meta_model.fit(stack_features, y_test)
        
        return rf, xgb, meta_model

    def _save_models(self, crop: str, rf_model, xgb_model, meta_model, scaler, features):
        """Persist models to disk"""
        version = datetime.datetime.now().strftime("%Y%m%d")
        joblib.dump(rf_model, f'models/rf_model_{crop.lower()}_{version}.pkl')
        joblib.dump(xgb_model, f'models/xgb_model_{crop.lower()}_{version}.pkl')
        joblib.dump(meta_model, f'models/meta_model_{crop.lower()}_{version}.pkl')
        joblib.dump(scaler, f'models/scaler_{crop.lower()}_{version}.pkl')
        joblib.dump(features, f'models/features_{crop.lower()}_{version}.pkl')

if __name__ == '__main__':
    # Example usage
    predictor = CropPricePredictor()
    df = predictor.load_data('groundnut4.csv', 'paddy4.csv')
    predictor.train_models(df)