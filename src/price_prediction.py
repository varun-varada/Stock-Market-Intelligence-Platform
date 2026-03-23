"""
Price Prediction Module
LSTM and XGBoost models for stock price prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """Base class for stock price prediction models."""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the predictor.
        
        Args:
            sequence_length: Number of time steps for sequence models
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, df: pd.DataFrame, feature_col: str = 'Close') -> Tuple:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with stock data
            feature_col: Column to use for prediction
            
        Returns:
            X, y arrays
        """
        data = df[feature_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM."""
        X = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
        return np.array(X)
    
    def predict_future(self, last_sequence: np.ndarray, days: int) -> np.ndarray:
        """Predict future prices."""
        predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_seq.reshape(1, -1), verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], pred[0, 0])
        
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


class LSTMPredictor(StockPredictor):
    """LSTM-based stock price predictor."""
    
    def __init__(self, sequence_length: int = 60):
        super().__init__(sequence_length)
        self.keras_model = None
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.keras_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.keras_model.compile(optimizer='adam', loss='mean_squared_error')
            return self.keras_model
        except ImportError:
            print("TensorFlow not available")
            return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model."""
        if self.keras_model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        self.keras_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return self.keras_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = X.reshape((X.shape[0], X.shape[1], 1))
        predictions = self.keras_model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions)


class XGBoostPredictor(StockPredictor):
    """XGBoost-based stock price predictor."""
    
    def __init__(self, sequence_length: int = 60):
        super().__init__(sequence_length)
        self.xgb_model = None
    
    def build_model(self):
        """Build XGBoost model."""
        try:
            import xgboost as xgb
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            return self.xgb_model
        except ImportError:
            print("XGBoost not available")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features including technical indicators.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with features
        """
        df = df.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        return df.dropna()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost model."""
        if self.xgb_model is None:
            self.build_model()
        
        self.xgb_model.fit(X_train, y_train)
        return self.xgb_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.xgb_model.predict(X)
        return predictions.reshape(-1, 1)


# Example usage
if __name__ == "__main__":
    # Example: Prepare dummy data
    print("Stock Prediction Module")
    print("Available models: LSTMPredictor, XGBoostPredictor")
