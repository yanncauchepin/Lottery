import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, binary_crossentropy, binary_focal_crossentropy
from keras.layers import Input
from sklearn.metrics import accuracy_score

def meta_modeling(df, size):
    
    X = df.iloc[:-size].values.astype(np.int32)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = df.iloc[size:].values.astype(np.int32)
    # Binary Crossentropy: y_ball = y_ball.reshape(y_ball.shape[0], y_ball.shape[1])
    y = y.reshape(y.shape[0], 1, y.shape[1])
    
    
    input_shape = (X.shape[1], X.shape[2])
    # Binary Crossentropy: output_ball_shape = (y_ball.shape[1],)
    output_shape = (y.shape[1], y.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=output_shape[1], activation='relu', return_sequences=False),
        Dense(units=output_shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=10)
    
    last_X = df.iloc[size:].values.astype(np.int32)
    last_X = last_X.reshape(last_X.shape[0], 1, last_X.shape[1])
    
    last_X_predicted = model.predict(last_X)
    pred = np.sum(last_X_predicted[-size:], axis=0)
    next_y = np.argpartition(pred, -size)[-size:]
    next_y = [y+1 for y in next_y]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Next values: {next_y}')
    
    return last_X_predicted, pred, next_y

    
    
    
    