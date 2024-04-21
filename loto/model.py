import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, binary_crossentropy, binary_focal_crossentropy
from keras.layers import Input
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/all_loto.csv', index_col=0)
one_hot_df = pd.read_csv('data/all_one_hot_loto.csv', index_col=0)
one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_loto.csv', index_col=0)
one_hot_star_df = pd.read_csv('data/all_one_hot_star_loto.csv', index_col=0)


def ball_modeling():
    
    X_ball = one_hot_ball_df.iloc[:-5].values.astype(np.int32)
    X_ball = X_ball.reshape(X_ball.shape[0], 1, X_ball.shape[1])
    y_ball = one_hot_ball_df.iloc[5:].values.astype(np.int32)
    # Binary Crossentropy: y_ball = y_ball.reshape(y_ball.shape[0], y_ball.shape[1])
    y_ball = y_ball.reshape(y_ball.shape[0], 1, y_ball.shape[1])
    
    
    input_ball_shape = (X_ball.shape[1], X_ball.shape[2])
    # Binary Crossentropy: output_ball_shape = (y_ball.shape[1],)
    output_ball_shape = (y_ball.shape[1], y_ball.shape[2])
    
    model = Sequential([
        Input(shape=input_ball_shape),
        LSTM(units=49, activation='relu', return_sequences=False),
        Dense(units=49, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_ball, y_ball, epochs=20, batch_size=10)
    
    last_ball = one_hot_ball_df.iloc[5:].values.astype(np.int32)
    last_ball = last_ball.reshape(last_ball.shape[0], 1, last_ball.shape[1])
    
    last_ball_predicted = model.predict(last_ball)
    pred = np.sum(last_ball_predicted[-5:], axis=0)
    next_ball = np.argpartition(pred, -5)[-5:]
    next_ball = [ball+1 for ball in next_ball]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Next balls: {next_ball}')
    
    return last_ball_predicted, pred, next_ball


def star_modeling():

    X_star = one_hot_star_df.iloc[:-1].values.astype(np.int32)
    X_star = X_star.reshape(X_star.shape[0], 1, X_star.shape[1])
    y_star = one_hot_star_df.iloc[1:].values.astype(np.int32)
    # Binary Crossentropy: y_star = y_star.reshape(y_star.shape[0], y_star.shape[1])
    y_star = y_star.reshape(y_star.shape[0], 1, y_star.shape[1])
    
    input_star_shape = (X_star.shape[1], X_star.shape[2])
    # Binary Crossentropy: output_star_shape = (y_star.shape[1], )
    output_star_shape = (y_star.shape[1], y_star.shape[2])
    
    model = Sequential([
        Input(shape=input_star_shape),
        LSTM(units=10, activation='relu', return_sequences=False),
        Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_star, y_star, epochs=20, batch_size=10)
    
    
    # import pdb ; pdb.set_trace()
    # y_star_predicted = model.predict(X_star)
    # for i in range(len(y_star_predicted)/2):
    #     pred = np.sum(y_star_predicted[2*i:2*(i+1)], axis=0)
    #     next_star = np.argpartition(pred, -2)[-2:]
    #     next_star = [star+1 for star in next_star]
    #     y_star_pred = [i+1 for i in range(12)]
    #     y_star_pred = [1 if i in next_star else 0 for i in y_star_pred]
    #     y_star_pred = np.array(y_star_pred)
    #     y_star_pred = y_star_pred.reshape(1, y_star_pred.shape[0])
    #     # evaluation_metric (y_star_pred, y_star[2i:2(i+1)])
        
    
    last_star = one_hot_star_df.iloc[:-1].values.astype(np.int32)
    last_star = last_star.reshape(last_star.shape[0], 1, last_star.shape[1])
    
    last_star_predicted = model.predict(last_star)
    pred = np.sum(last_star_predicted[-1:], axis=0)
    next_star = np.argpartition(pred, -1)[-1:]
    next_star = [star+1 for star in next_star]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Next stars: {next_star}')
    
    return last_star_predicted, pred, next_star


if __name__=='__main__':
    
    predictions_ball = ball_modeling()
    predictions_star = star_modeling()
    
    
    
    