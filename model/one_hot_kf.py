import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

class KF():

    def __init__(self, numbers, x, P):
        self.numbers = numbers
        self.x = x
        self.x_prior = np.copy(self.x)
        self.P = P
        self.P_prior = np.copy(self.P)
        self.F = np.diag([1/self.numbers]*self.numbers)
        self.H = np.eye(numbers)
        self.Q = np.diag([0.2]*numbers)
        self.R = np.diag([0]*numbers) 

    def softmax_proba(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def predict(self):
        self.x = np.dot(self.F, self.x_prior)
        self.x_= self.softmax_proba(self.x) + 0.01*(self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        # self.P = np.dot(np.dot(self.F, self.P_prior), self.F.T) + self.Q

    def update(self, z):
        self.y = z - np.dot(self.H, self.x)
        print(self.H.shape)
        print(self.P.shape)
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.y)
        # self.P = self.P - np.dot(np.dot(self.K, self.H), self.P)
        self.x_= self.softmax_proba(self.x) + 0.01*(self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        # self.x_ = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        # self.P_prior = np.copy(self.P)

        self.x_prior = np.copy(self.x) 

def categorical_crossentropy(y_true, y_pred):
    # Convert to numpy arrays in case they're lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0) error
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    
    # Compute categorical cross-entropy
    cross_entropy = -np.sum(y_true * np.log(y_pred))
    return cross_entropy

def meta_modeling(df, size, numbers):
    files = glob.glob('history_kalman_filter/{sub}/*')
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    x = np.array([1/numbers]*numbers)
    P = np.cov(df, rowvar=False)
    kalman_filter = KF(numbers, x, P)
    i = 0
    for date, draw in df.iloc[-20:].iterrows():
        i += 1
        kalman_filter.predict()
        kalman_filter.update(np.diag([draw]))
        print(f'{date}: {categorical_crossentropy(draw, kalman_filter.x_)}')
        plt.figure()
        plt.bar(range(len(draw)), draw*2*np.max(kalman_filter.x_)) 
        plt.bar(range(len(kalman_filter.x_)), kalman_filter.x_)
        plt.savefig('history_kalman_filter/{}.png'.format(date))
    kalman_filter.predict()
    proba = pd.DataFrame(kalman_filter.x_, index=range(1, numbers+1))
    proba = proba.sort_values(by=0, ascending=False)
    return proba

if __name__ == '__main__':
    df = pd.read_csv('data/concat_all_one_hot_ball_loto.csv', index_col=0)
    print(meta_modeling(df, 5, 49))
