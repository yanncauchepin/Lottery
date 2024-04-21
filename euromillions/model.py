import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, binary_crossentropy, binary_focal_crossentropy
from keras.layers import Input

df = pd.read_csv('data/all_euromillions.csv', index_col=0)
one_hot_df = pd.read_csv('data/all_one_hot_euromillions.csv', index_col=0)
one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)


def integer_basic_modeling():

    X = df.iloc[:-2].values.astype(np.int32)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = df.iloc[1:-1].values.astype(np.int32)
    y = y.reshape(y.shape[0], y.shape[1], 1)
    
    input_shape = (X.shape[1], X.shape[2])
    output_shape = (y.shape[1], y.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, activation='relu'),
        Dense(units=output_shape[0]) #, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X, y, epochs=50, batch_size=10)
    
    
    last = df.iloc[2:].values.astype(np.int32)
    last = last.reshape(last.shape[0], last.shape[1], 1)
    
    next_ = model.predict(last)
    print(next_[-1])
    
    return next_, next_[-1]


def one_hot_ball_modeling():
    
# =============================================================================
#     # MSE version
# 
#     X_ball = one_hot_ball_df.iloc[:-10].values.astype(np.int32)
#     X_ball = X_ball.reshape(X_ball.shape[0], 1, X_ball.shape[1])
#     y_ball = one_hot_ball_df.iloc[5:-5].values.astype(np.int32)
#     y_ball = y_ball.reshape(y_ball.shape[0], 1, y_ball.shape[1])
#     
#     
#     input_ball_shape = (X_ball.shape[1], X_ball.shape[2])
#     output_ball_shape = (y_ball.shape[1], y_ball.shape[2])
#     
#     model = Sequential([
#         Input(shape=input_ball_shape),
#         LSTM(units=50, activation='relu', return_sequences=False, seed=30061997)
#     ])
#     
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     
#     model.fit(X_ball, y_ball, epochs=25, batch_size=10)
#     
#     
#     last_ball = one_hot_ball_df.iloc[10:].values.astype(np.int32)
#     last_ball = last_ball.reshape(last_ball.shape[0], 1, last_ball.shape[1])
#     
#     predictions = model.predict(last_ball)
#     
#     pred = np.sum(predictions[-5:], axis=0)
#     next_ball = np.argpartition(pred, -5)[-5:]
#     next_ball = [ball+1 for ball in next_ball]
#     pred = pd.DataFrame(pred)
#     pred.index += 1
#     pred = pred.sort_values(by=0, ascending=False)
#     print(f'MSE: {next_ball}')
#     
#     # return predictions, pred, next_ball
# =============================================================================

    # Binary Crossentropy version

    X_ball = one_hot_ball_df.iloc[:-10].values.astype(np.int32)
    X_ball = X_ball.reshape(X_ball.shape[0], 1, X_ball.shape[1])
    y_ball = one_hot_ball_df.iloc[5:-5].values.astype(np.int32)
    y_ball = y_ball.reshape(y_ball.shape[0], y_ball.shape[1])
    
    
    input_ball_shape = (X_ball.shape[1], X_ball.shape[2])
    output_ball_shape = (y_ball.shape[1],)
    
    model = Sequential([
        Input(shape=input_ball_shape),
        LSTM(units=50, activation='relu', return_sequences=False, seed=30061997)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_ball, y_ball, epochs=25, batch_size=10)
    
    
    last_ball = one_hot_ball_df.iloc[5:-5].values.astype(np.int32)
    last_ball = last_ball.reshape(last_ball.shape[0], 1, last_ball.shape[1])
    
    predictions = model.predict(last_ball)
    
    pred = np.sum(predictions[-5:], axis=0)
    next_ball = np.argpartition(pred, -5)[-5:]
    next_ball = [ball+1 for ball in next_ball]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Binary Crossentropy: {next_ball}')
    
    return predictions, pred, next_ball


def one_hot_star_modeling():
    
# =============================================================================
#     # MSE version
# 
#     X_star = one_hot_star_df.iloc[:-10].values.astype(np.int32)
#     X_star = X_star.reshape(X_star.shape[0], 1, X_star.shape[1])
#     y_star = one_hot_star_df.iloc[5:-5].values.astype(np.int32)
#     y_star = y_star.reshape(y_star.shape[0], 1, y_star.shape[1])
#     
#     
#     input_star_shape = (X_star.shape[1], X_star.shape[2])
#     output_star_shape = (y_star.shape[1], y_star.shape[2])
#     
#     model = Sequential([
#         Input(shape=input_star_shape),
#         LSTM(units=12, activation='relu', return_sequences=False, seed=30061997)
#     ])
#     
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     
#     model.fit(X_star, y_star, epochs=25, batch_size=10)
#     
#     
#     last_star = one_hot_star_df.iloc[10:].values.astype(np.int32)
#     last_star = last_star.reshape(last_star.shape[0], 1, last_star.shape[1])
#     
#     predictions = model.predict(last_star)
#     
#     pred = np.sum(predictions[-5:], axis=0)
#     next_star = np.argpartition(pred, -2)[-2:]
#     next_star = [star+1 for star in next_star]
#     pred = pd.DataFrame(pred)
#     pred.index += 1
#     pred = pred.sort_values(by=0, ascending=False)
#     print(f'MSE: {next_star}')
#     
#     # return predictions, pred, next_star
# =============================================================================

    # Binary Crossentropy version

    X_star = one_hot_star_df.iloc[:-10].values.astype(np.int32)
    X_star = X_star.reshape(X_star.shape[0], 1, X_star.shape[1])
    y_star = one_hot_star_df.iloc[5:-5].values.astype(np.int32)
    y_star = y_star.reshape(y_star.shape[0], y_star.shape[1])
    
    
    input_star_shape = (X_star.shape[1], X_star.shape[2])
    output_star_shape = (y_star.shape[1], )
    
    model = Sequential([
        Input(shape=input_star_shape),
        LSTM(units=12, activation='relu', return_sequences=False, seed=30061997)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_star, y_star, epochs=25, batch_size=10)
    
    
    last_star = one_hot_star_df.iloc[5:-5].values.astype(np.int32)
    last_star = last_star.reshape(last_star.shape[0], 1, last_star.shape[1])
    
    predictions = model.predict(last_star)
    
    pred = np.sum(predictions[-5:], axis=0)
    next_star = np.argpartition(pred, -2)[-2:]
    next_star = [star+1 for star in next_star]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Binary Crossentropy: {next_star}')
    
    return predictions, pred, next_star


if __name__=='__main__':
    
    # predictions = integer_basic_modeling()
    
    predictions_ball = one_hot_ball_modeling()
    predictions_star = one_hot_star_modeling()
    
    
    
    