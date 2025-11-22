import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Dict, Any, List

from modules.data_building import main as load_data

class LotteryPredictorLSTM:
    """
    A Deep Learning class for predicting temporal lottery draws using an LSTM model.
    It uses one-hot encoded draw data and Binary Cross-Entropy loss to avoid
    frequency bias.
    """
    def __init__(self, sequence_length: int, k_drawn: int, lstm_units: int = 128, dropout_rate: float = 0.3):
        """
        Initializes the predictor with model hyper-parameters.

        Args:
            sequence_length: The number of previous draws (L) to use for prediction.
            k_drawn: The number of balls (K) drawn in the lottery (e.g., 5 or 6).
            lstm_units: Number of units in the LSTM layer.
            dropout_rate: Dropout rate for regularization.
        """
        self.SEQUENCE_LENGTH = sequence_length
        self.K_DRAWN = k_drawn
        self.LSTM_UNITS = lstm_units
        self.DROPOUT_RATE = dropout_rate
        self.N_BALLS = None 
        self.model: Sequential = None
        
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms the flat array of draws into temporal sequences (X) and targets (y).
        
        Args:
            data: A 2D NumPy array of one-hot encoded draws (draws, N_BALLS).
            
        Returns:
            A tuple (X, y) where X is the input sequence and y is the target draw.
        """
        X, y = [], []
        seq_length = self.SEQUENCE_LENGTH
        
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array (draws, N_BALLS).")
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_data(self, df_one_hot: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Prepares and splits the dataset for training.
        
        Args:
            df_one_hot: The concatenated one-hot encoded DataFrame (dates as index).
            test_size: Proportion of the data to be used as test set.
            random_state: Seed for the random split.
        """
        self.N_BALLS = df_one_hot.shape[1]
        all_draws_array = df_one_hot.values
        
        # print(f"--- Data Configuration ---")
        # print(f"Total draws loaded: {len(all_draws_array)}")
        # print(f"Total possible balls (N): {self.N_BALLS}")
        # print(f"Sequence Length (L): {self.SEQUENCE_LENGTH}")
        
        X, y = self.create_sequences(all_draws_array)
        
        # print(f"Total training samples created: {X.shape[0]}")
        # print(f"X shape (Samples, Seq_Length, N_Balls): {X.shape}")
        # print(f"y shape (Samples, N_Balls): {y.shape}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        # print(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")


    def build_model(self):
        """Defines and compiles the BiLSTM prediction model."""
        if self.N_BALLS is None:
            raise ValueError("Data must be prepared before building the model. Call prepare_data() first.")
            
        model = Sequential([
            Bidirectional(LSTM(units=self.LSTM_UNITS, activation='tanh'), 
                 input_shape=(self.SEQUENCE_LENGTH, self.N_BALLS)),
            Dropout(self.DROPOUT_RATE),
            Dense(self.N_BALLS, activation='sigmoid') 
        ])
        
        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.Precision(thresholds=0.5)]) 
        
        self.model = model
        # print("--- Model Summary ---")
        # self.model.summary()
        # print("-" * 30)

    def train_model(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Trains the built LSTM model.
        
        Returns:
            The Keras history dictionary.
        """
        if self.model is None or self.X_train is None:
            raise ValueError("Model must be built and data prepared before training.")
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
            
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,             
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history.history

    def predict_next_draw(self, input_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Predicts the next draw by selecting the K balls with the highest probabilities.
        
        Args:
            input_sequence: A single sequence of shape (SEQUENCE_LENGTH, N_BALLS).
            
        Returns:
            A dictionary containing the predicted balls (1-indexed), their probabilities, 
            and the full probability vector.
        """
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
            
        input_seq = np.expand_dims(input_sequence, axis=0)
        
        probabilities = self.model.predict(input_seq, verbose=0)[0]

        predicted_ball_indices = np.argsort(probabilities)[-self.K_DRAWN:][::-1]

        predicted_balls = predicted_ball_indices + 1
        
        top_k_probabilities = probabilities[predicted_ball_indices]
        
        return {
            'predictions': predicted_balls, 
            'top_k_probabilities': top_k_probabilities, 
            'full_probabilities': probabilities
        }

if __name__ == '__main__':

    N_BALLS_EUROMILLIONS = 50 
    K_DRAWN_EUROMILLIONS = 5
    
    all_one_hot_df = load_data('euromillions')
    
    df_concat_ball_df = all_one_hot_df['ball_df']
    
    SEQUENCE_LENGTH = 10 
    EPOCHS = 20 
    
    predictor = LotteryPredictorLSTM(
        sequence_length=SEQUENCE_LENGTH, 
        k_drawn=K_DRAWN_EUROMILLIONS
    )
    
    predictor.prepare_data(df_concat_ball_df)

    predictor.build_model()
    
    predictor.train_model(epochs=EPOCHS)
    
    input_seq_for_prediction = predictor.X_test[0]
    
    prediction_result = predictor.predict_next_draw(input_seq_for_prediction)

    print("-" * 30)
    print("--- Final Prediction ---")
    print(f"Input Sequence Shape: {input_seq_for_prediction.shape}")
    print(f"Predicted Draw (Ball Numbers): {prediction_result['predictions']}")
    print(f"Corresponding Probabilities: {np.round(prediction_result['top_k_probabilities'], 4)}")
    print("-" * 30)