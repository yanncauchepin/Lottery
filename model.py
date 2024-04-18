import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv('build_dataframe.csv', index_col=0)

X = df.iloc[:, :-1].values.astype(np.int32)
y = df.iloc[:, -1].values.astype(np.int32)

X = X.reshape(X.shape[0], 1, X.shape[1])

# Define LSTM model
model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(1, X.shape[2])),
    Dense(units=1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=50, batch_size=32)

# Make predictions (pas de test car on a déjà utilisé toutes les données pour l'entraînement)
predictions = model.predict(X)