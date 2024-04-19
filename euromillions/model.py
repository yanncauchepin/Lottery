import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv('all_euromillions.csv', index_col=0)

X = df.iloc[:, :-1].values.astype(np.int32)
y = df.iloc[:, -1].values.astype(np.int32)

X = X.reshape(X.shape[0], 1, X.shape[1])

model = Sequential([
    LSTM(units=100, activation='relu', input_shape=(1, X.shape[2])),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=200, batch_size=32)


last = df.iloc[:, 1:].values.astype(np.int32)
last = last.reshape(last.shape[0], 1, last.shape[1])

predictions = model.predict(last)
print(predictions)