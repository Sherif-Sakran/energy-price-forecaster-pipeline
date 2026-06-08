import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to transform and encode tabular power features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config["features"]
        self.target_col = self.config["target_col"]
        self.country_code = self.config["country_code"]
        self.lags = self.config["lags"]
        self.native_categorical_cols = self.config["native_categorical_cols"]
        self.features_to_drop = self.config["features_to_drop"]

    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies residual load, datetime extractions, and drops features."""
        df_feat = df.copy()

        # 1. Residual Load
        df_feat['residual_load'] = (
            df_feat["Forecasted Load"] - 
            df_feat["Solar"] - 
            df_feat["Wind Offshore"] - 
            df_feat["Wind Onshore"]
        )

        # 2. Datetime Features
        df_feat['hour'] = df_feat.index.hour
        df_feat['day_of_week'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        df_feat['quarter'] = df_feat.index.quarter
        df_feat['day_of_month'] = df_feat.index.day
        df_feat['day_of_year'] = df_feat.index.dayofyear

        # Weekend Flag
        df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Drop specific features (e.g., day_of_year)
        existing_drops = [col for col in self.features_to_drop if col in df_feat.columns]
        if existing_drops:
            df_feat = df_feat.drop(columns=existing_drops)

        return df_feat

    def create_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates historical price lag features and drops resulting NaNs."""
        df_feat = df.copy()
        
        for lag in self.lags:
            col_name = f'price_lag_{lag}h'
            target_series = df_feat[f'price_{self.country_code.lower()}']
            df_feat[col_name] = target_series.shift(lag)

        # Notebook explicitly drops NAs based on the 168h lag constraint
        df_feat.dropna(subset=['price_lag_168h'], inplace=True)
        return df_feat

    def apply_categorical_and_cyclical_encodings(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Applies continuous cyclical encodings (sin/cos) and tags categorical columns.
        Returns X, y, and the list of active categorical features.
        """
        df_feat = df.copy()
        
        if 'is_holiday' in df_feat.columns:
            df_feat['is_holiday'] = df_feat['is_holiday'].astype(int)

        # 1. Cyclical Feature Engineering (MUST happen before categorical casting)
        if 'hour' in df_feat.columns:
            df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
            df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
            
        if 'day_of_week' in df_feat.columns:
            df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7.0)
            df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7.0)
            
        if 'month' in df_feat.columns:
            df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12.0)
            df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12.0)

        # 2. Native Categorical Encodings for LightGBM
        for col in self.native_categorical_cols:
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].astype('category')

        feature_cols = [col for col in df_feat.columns if col != self.target_col]
        
        return df_feat[feature_cols], df_feat[self.target_col], self.native_categorical_cols

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline runner for the feature engineering class."""
        df = df.sort_index()
        df.index.name = 'date'
        
        df = self.create_base_features(df)
        df = self.create_lags(df)
        return df