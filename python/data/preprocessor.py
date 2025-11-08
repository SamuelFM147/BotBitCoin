"""
Data preprocessing pipeline for Bitcoin market data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing for time series financial data
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_columns = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        """
        logger.info("Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method for price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        logger.info(f"Data cleaned. Remaining samples: {len(df)}")
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: DataFrame with features
            fit: If True, fit the scaler; otherwise use existing scaler
        """
        if self.feature_columns is None:
            # Exclude timestamp and target columns
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['timestamp', 'target', 'date']]
        
        if fit:
            logger.info("Fitting scaler...")
            scaled_values = self.scaler.fit_transform(df[self.feature_columns])
        else:
            scaled_values = self.scaler.transform(df[self.feature_columns])
        
        # Create normalized dataframe
        df_normalized = df.copy()
        df_normalized[self.feature_columns] = scaled_values
        
        return df_normalized
    
    def create_sequences(self, 
                        df: pd.DataFrame, 
                        lookback: int = 50,
                        target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with features
            lookback: Number of time steps to look back
            target_col: Column name for target variable
            
        Returns:
            X: Array of sequences (samples, timesteps, features)
            y: Array of targets (samples,)
        """
        logger.info(f"Creating sequences with lookback={lookback}")
        
        features = df[self.feature_columns].values
        targets = df[target_col].values if target_col in df.columns else None
        
        X, y = [], []
        
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            if targets is not None:
                y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y) if targets is not None else None
        
        logger.info(f"Created {len(X)} sequences")
        return X, y
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: Optional[np.ndarray] = None,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> dict:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature sequences
            y: Target values (optional)
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            
        Returns:
            Dictionary with train, val, and test splits
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        splits = {
            'X_train': X[:train_end],
            'X_val': X[train_end:val_end],
            'X_test': X[val_end:]
        }
        
        if y is not None:
            splits.update({
                'y_train': y[:train_end],
                'y_val': y[train_end:val_end],
                'y_test': y[val_end:]
            })
        
        logger.info(f"Data split - Train: {len(splits['X_train'])}, "
                   f"Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
        
        return splits
    
    def preprocess_pipeline(self, 
                           file_path: str,
                           lookback: int = 50,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> dict:
        """
        Complete preprocessing pipeline
        
        Returns:
            Dictionary with preprocessed and split data
        """
        # Load and clean data
        df = self.load_data(file_path)
        df = self.clean_data(df)
        
        # Normalize
        df_normalized = self.normalize_data(df, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(df_normalized, lookback=lookback)
        
        # Split data
        splits = self.split_data(X, y, train_ratio, val_ratio)
        
        # Add original dataframe for reference
        splits['df_original'] = df
        splits['df_normalized'] = df_normalized
        
        return splits


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # This would process your data
    # data = preprocessor.preprocess_pipeline('data/bitcoin_historical.csv')
