import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

SEQUENCE = 1000

def softmax_proba(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

def binary_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return cross_entropy

def categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred))
    return cross_entropy

def compute_metric(y_pred):
    y_pred = np.array(y_pred)
    return softmax_proba(y_pred) + 0.01*(y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

class AdjustProbabilities(Layer):
    def __init__(self, adjustment_weights, **kwargs):
        super(AdjustProbabilities, self).__init__(**kwargs)
        self.adjustment_weights = adjustment_weights

    def call(self, inputs):
        y_pred = inputs
        adjusted_output = tf.add(y_pred, self.adjustment_weights)
        return tf.nn.softmax(adjusted_output)

def meta_modeling(lottery, df, size, numbers):

    X = df.iloc[:-1].values
    mean = 1- np.mean(df, axis=0)
    mean_ = (mean - np.min(mean))/(np.max(mean)-np.min(mean))
    adjustement_weights = softmax_proba(mean_)
    y = df.iloc[-1].values

    sequence = len(df)-1

    X_ = X[:-1]
    X_ = X_.reshape(1, X_.shape[0], X_.shape[1])
    y_ = X[-1].reshape(1, -1)

    X_last = X[1:]
    X_last = X_last.reshape(1, X_last.shape[0], X_last.shape[1])
    y_last = y.reshape(1, -1) 

    model = Sequential([
            LSTM(128, input_shape=(X_.shape[1], X_.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(y_.shape[1], activation='softmax'),
            AdjustProbabilities(adjustement_weights)
        ])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        min_delta=0.001,     
        patience=5,          
        restore_best_weights=True  
    )

    history = model.fit(
        X_, 
        y_, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_last, y_last),
        callbacks=[early_stopping]
    )

    print(f'{binary_crossentropy(y_, compute_metric(model.predict(X_)))}')
    print(f'{binary_crossentropy(y, compute_metric(model.predict(X_last)))}')
    
    proba = pd.DataFrame(compute_metric(model.predict(X_last))[0], index=range(1, numbers+1))
    proba = proba.sort_values(by=0, ascending=False)
    
    return proba

    

if __name__ == '__main__':
    df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    result = meta_modeling('euromillions_ball', df, 5, 50)
    print(result)
    
    
    