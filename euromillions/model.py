import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

df = pd.read_csv('all_euromillions.csv', index_col=0)
one_hot_df = pd.read_csv('all_one_hot_euromillions.csv', index_col=0)
one_hot_ball_df = pd.read_csv('all_one_hot_ball_euromillions.csv', index_col=0)
one_hot_star_df = pd.read_csv('all_one_hot_star_euromillions.csv', index_col=0)


# =============================================================================
# def integer_basic_modeling():
# 
#     X = df.iloc[:-2].values.astype(np.int32)
#     X = X.reshape(X.shape[0], X.shape[1], 1)
#     y = df.iloc[-2].values.astype(np.int32)
#     y = y.reshape(1, y.shape[0], 1)
#     
#     input_shape = (X.shape[1], X.shape[2])
#     output_shape = (y.shape[1], y.shape[2])
#     
#     model = Sequential([
#         LSTM(units=100, activation='relu', input_shape=input_shape),
#         Dense(units=output_shape[1]) #, activation='softmax')
#     ])
#     
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     
#     model.fit(X, y, epochs=50, batch_size=70)
#     
#     
#     last = df.iloc[1:-1].values.astype(np.int32)
#     last = last.reshape(last.shape[1], last.shape[0], 1)
#     
#     predictions = model.predict(last)
#     print(predictions)
#     
#     return predictions
# =============================================================================

# =============================================================================
# def second_version():
#     
#     def metric_ball(y_true, y_pred):
#         
#         y_pred_rounded =  tf.cast(tf.round(y_pred), tf.int32)
#         errors = 0
#         y_true_cast = tf.cast(y_true, tf.int32)
#         
#         # def compute_errors(true, pred_rounded):
#         #     errors = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.reduce_any(tf.equal(true, pred_rounded))), tf.float32))
#         #     return errors
# 
#         # errors = tf.map_fn(lambda x: compute_errors(x, y_pred_rounded), (y_true_cast[0], ), dtype=tf.float32)
#         # errors = tf.reduce_sum(errors)
#         errors+=1/50*0.1*tf.keras.losses.mean_squared_error(y_true, y_pred)
#         
#         return tf.reduce_sum(errors)
#     
#     X_ball = df.loc[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']]
#     y_ball = X_ball.iloc[:, -2] ; X_ball.drop(X_ball.columns[[-1,-2]], axis=1, inplace=True)
#     X_ball = X_ball.values.astype(int)
#     y_ball = y_ball.values.astype(int)
#     X_ball = X_ball.reshape(X_ball.shape[0], 1, X_ball.shape[1])
#     
#     model_ball = Sequential([
#         LSTM(
#             units=200, 
#             activation='relu', 
#             input_shape=(1, X_ball.shape[2]),
#             ),
#         Dense(units=1)
#     ])
#     
#     model_ball.compile(optimizer='adam', loss=metric_ball)
#     model_ball.fit(X_ball, y_ball, epochs=200, batch_size=20)
#     
#     def metric_star(y_true, y_pred):
#         
#         y_pred_rounded =  tf.cast(tf.round(y_pred), tf.int32)
#         errors = 0
#         y_true_cast = tf.cast(y_true, tf.int32)
#         
#         # def compute_errors(true, pred_rounded):
#         #     errors = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.reduce_any(tf.equal(true, pred_rounded))), tf.float32))
#         #     return errors
# 
#         # errors = tf.map_fn(lambda x: compute_errors(x, y_pred_rounded), (y_true_cast[0], ), dtype=tf.float32)
#         # errors = tf.reduce_sum(errors)
#         errors+=1/12*0.1*tf.keras.losses.mean_squared_error(y_true, y_pred)
#         
#         return tf.reduce_sum(errors)
#     
#     X_ball_last = df.loc[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']]
#     X_ball_last.drop(X_ball_last.columns[[0, -1]], axis=1, inplace=True)
#     X_ball_last = X_ball_last.values.astype(int)
#     X_ball_last = X_ball_last.reshape(X_ball_last.shape[0], 1, X_ball_last.shape[1])
#     next_ball = model_ball.predict(X_ball_last)
#     
#     X_star = df.loc[['star_1','star_2']]
#     y_star = X_star.iloc[:, -2] ; X_star.drop(X_star.columns[[-1, -2]], axis=1, inplace=True)
#     X_star = X_star.values.astype(int)
#     y_star = y_star.values.astype(int)
#     X_star = X_star.reshape(X_star.shape[0], 1, X_star.shape[1])
#     
#     model_star = Sequential([
#         LSTM(
#             units=200, 
#             activation='relu', 
#             input_shape=(1, X_star.shape[2]),
#             ),
#         Dense(units=1)
#     ])
#     
#     model_star.compile(optimizer='adam', loss=metric_star)
#     model_star.fit(X_star, y_star, epochs=200, batch_size=32)
#     
#     X_star_last = df.loc[['star_1','star_2']]
#     X_star_last.drop(X_star_last.columns[[0, -1]], axis=1, inplace=True)
#     X_star_last = X_star_last.values.astype(int)
#     X_star_last = X_star_last.reshape(X_star_last.shape[0], 1, X_star_last.shape[1])
#     next_star = model_star.predict(X_star_last)
#     
#     return (next_ball, next_star)
# =============================================================================
    
def one_hot_modeling():
    
    X_ball = one_hot_ball_df.iloc[:-10].values.astype(np.int32)
    X_ball = X_ball.reshape(X_ball.shape[1], X_ball.shape[0], 1)
    y_ball = one_hot_ball_df.iloc[-10:-5].values.astype(np.int32)
    y_ball = y_ball.reshape(y_ball.shape[1], y_ball.shape[0], 1)
    
    model = Sequential([
        LSTM(
            units=200, 
            activation='relu', 
            input_shape=(X_ball.shape[1], X_ball.shape[2]),
            ),
        Dense(units=5)
    ])
    
    model.compile(optimizer='adam', loss=CategoricalCrossentropy())
    
    model.fit(X_ball, y_ball, epochs=25, batch_size=20)
    
    
    last_ball = one_hot_ball_df.iloc[-5:].values.astype(np.int32)
    last_ball = last_ball.reshape(last_ball.shape[1], last_ball.shape[0], 1)
    
    next_ball = model.predict(last_ball)
    print(next_ball)
    return next_ball    

if __name__=='__main__':
    predictions = one_hot_modeling()
    
    
    
    