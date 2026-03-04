import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.features.transformers import SolarWindTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarWindPreprocessor:
    """
    Main preprocessing pipeline for solar wind data.
    Handles feature engineering, selection, scaling, and validation.
    """
    
    # Final selected features (from feature_selection.ipynb)
    FINAL_FEATURES = [
        'DOY',
        'Hour',
        'IMF_Mag',
        'Plasma_Speed',
        'Proton_Density',
        'Bx_GSE',
        'By_GSE',
        'Bz_GSE',
        'Energy_Flux',
        'Energy_3h_Avg',
        'B_total',
        'B_azimuth'
    ]
    
    TARGET = 'Kp_index'
    
    def __init__(self, scaler_type='standard', fit_scaler=True):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard' or 'robust' (robust is better for outliers)
            fit_scaler: Whether to fit scalers during preprocessing
        """
        self.scaler_type = scaler_type
        self.fit_scaler = fit_scaler
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        logger.info(f"Initialized SolarWindPreprocessor with {scaler_type} scaler")
    
    def engineer_features(self, df):
        """
        Apply feature engineering transformations.
        
        Args:
            df: Raw DataFrame with columns: Plasma_Speed, Proton_Density, Bx_GSE, By_GSE, Bz_GSE, Kp_index
        
        Returns:
            DataFrame with engineered features added
        """
        logger.info("Engineering features...")
        df = df.copy()
        
        # Apply all transformations
        df = SolarWindTransformer.transform(
            df,
            speed_col='Plasma_Speed',
            bz_col='Bz_GSE',
            time_col='time_tag' if 'time_tag' in df.columns else 'Impact_Time'
        )
        
        logger.info(f"Features after engineering: {df.shape[1]}")
        return df
    
    def select_features(self, df, include_target=True):
        """
        Select only the finals features from the dataset.
        
        Args:
            df: DataFrame with all features
            include_target: Whether to include target variable
        
        Returns:
            DataFrame with selected features only
        """
        logger.info(f"Selecting {len(self.FINAL_FEATURES)} features...")
        
        # Check which features are available
        available_features = [f for f in self.FINAL_FEATURES if f in df.columns]
        missing_features = set(self.FINAL_FEATURES) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        selected = df[available_features].copy()
        
        if include_target and self.TARGET in df.columns:
            selected[self.TARGET] = df[self.TARGET]
        
        logger.info(f"Selected features shape: {selected.shape}")
        return selected
    
    def handle_missing_values(self, df, method='interpolate', limit=3):
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential NaN values
            method: 'interpolate', 'drop', or 'forward_fill'
            limit: Max consecutive values to interpolate
        
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with method: {method}")
        
        initial_NaN_count = df.isna().sum().sum()
        
        if method == 'interpolate':
            df = df.interpolate(method='linear', limit=limit)
            # Drop remaining NaN
            df = df.dropna()
        elif method == 'drop':
            df = df.dropna()
        elif method == 'forward_fill':
            df = df.fillna(method='ffill', limit=limit)
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        final_NaN_count = df.isna().sum().sum()
        logger.info(f"Missing values reduced from {initial_NaN_count} to {final_NaN_count}")
        
        return df
    
    def remove_outliers(self, df, method='iqr', threshold=1.5, target_only=False):
        """
        Remove outliers from the dataset.
        
        Args:
            df: DataFrame
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: IQR multiplier (1.5) or z-score threshold (3)
            target_only: Only flag outliers in target variable
        
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        
        initial_rows = len(df)
        
        if target_only:
            features_to_check = [self.TARGET] if self.TARGET in df.columns else []
        else:
            features_to_check = [col for col in df.columns if col != self.TARGET]
        
        if method == 'iqr':
            for col in features_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[features_to_check], nan_policy='omit'))
            df = df[(z_scores < threshold).all(axis=1)]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} outlier rows ({100*removed_rows/initial_rows:.2f}%)")
        
        return df
    
    def scale_features(self, X, fit=True):
        """
        Scale features using the configured scaler.
        
        Args:
            X: Feature matrix (excluding target)
            fit: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            Scaled feature matrix
        """
        if fit:
            logger.info(f"Fitting {self.scaler_type} scaler...")
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                logger.warning("Scaler not fitted! Call with fit=True first on training data.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df, fit_scaler=True, handle_missing='interpolate', 
                   remove_outliers_method=None, scale=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            fit_scaler: Whether to fit the scaler (True for training data)
            handle_missing: Method to handle NaN ('interpolate', 'drop', 'forward_fill')
            remove_outliers_method: 'iqr', 'zscore', or None
            scale: Whether to scale features
        
        Returns:
            Tuple of (X_scaled, y) if target present, else (X_scaled, None)
        """
        logger.info("=== STARTING PREPROCESSING PIPELINE ===")
        
        df = df.copy()
        
        # 1. Engineer features
        df = self.engineer_features(df)
        
        # 2. Select features
        df = self.select_features(df, include_target=True)
        
        # 3. Handle missing values
        df = self.handle_missing_values(df, method=handle_missing)
        
        # 4. Remove outliers (optional)
        if remove_outliers_method:
            df = self.remove_outliers(df, method=remove_outliers_method)
        
        # 5. Separate features and target
        if self.TARGET in df.columns:
            X = df[[col for col in df.columns if col != self.TARGET]]
            y = df[self.TARGET]
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        else:
            X = df
            y = None
            logger.info(f"Features shape: {X.shape}, No target found")
        
        # 6. Scale features
        if scale:
            X_scaled = self.scale_features(X, fit=fit_scaler)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info(f"Features scaled using {self.scaler_type} scaler")
        else:
            X_scaled = X
        
        logger.info("=== PREPROCESSING COMPLETE ===\n")
        
        return X_scaled, y
    
    def preprocess_inference(self, df):
        """
        Preprocess data for inference (uses fitted scaler, no target required).
        
        Args:
            df: Raw input DataFrame
        
        Returns:
            X_scaled: Preprocessed feature matrix
        """
        if not self.is_fitted:
            logger.warning("Scaler not fitted! Results may be incorrect.")
        
        return self.preprocess(df, fit_scaler=False, handle_missing='interpolate', 
                              remove_outliers_method=None, scale=True)[0]
    
    def get_feature_names(self):
        """Return the list of final feature names."""
        return self.FINAL_FEATURES
    
    def get_scaler(self):
        """Return the fitted scaler object."""
        return self.scaler if self.is_fitted else None


class DataValidator:
    """
    Validates data quality and consistency before/after preprocessing.
    """
    
    @staticmethod
    def validate_features(df, required_features):
        """
        Check if all required features are present.
        
        Args:
            df: DataFrame
            required_features: List of feature names
        
        Returns:
            Boolean indicating if all required features present
        """
        missing = set(required_features) - set(df.columns)
        if missing:
            logger.error(f"Missing features: {missing}")
            return False
        logger.info(f"✓ All {len(required_features)} required features present")
        return True
    
    @staticmethod
    def validate_no_nans(df, strict=False):
        """
        Check for NaN values.
        
        Args:
            df: DataFrame
            strict: If True, fail on any NaN; if False, just warn
        
        Returns:
            Boolean indicating data quality
        """
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            msg = f"Found {nan_count} NaN values"
            if strict:
                logger.error(msg)
                return False
            else:
                logger.warning(msg)
        else:
            logger.info("✓ No NaN values found")
        return True
    
    @staticmethod
    def validate_data_stats(df):
        """
        Print summary statistics for validation.
        
        Args:
            df: DataFrame
        """
        logger.info("\n--- Data Statistics ---")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        logger.info(f"\nDescriptive stats:\n{df.describe()}")