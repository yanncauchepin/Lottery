import numpy as np
import pandas as pd
from euromillions.data_building import main as euromillions_data
from loto.data_building import main as loto_data
from model.one_hot_lstm import meta_modeling as one_hot_lstm


def loto():
    
    loto_data()
    
    # df = pd.read_csv('data/all_loto.csv', index_col=0)
    # one_hot_df = pd.read_csv('data/all_one_hot_loto.csv', index_col=0)
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_loto.csv', index_col=0)

    predictions_ball = one_hot_lstm(one_hot_ball_df, 5)
    predictions_star = one_hot_lstm(one_hot_star_df, 1)

    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}



def euromillions():
    
    euromillions_data()
    
    # df = pd.read_csv('data/all_euromillions.csv', index_col=0)
    # one_hot_df = pd.read_csv('data/all_one_hot_euromillions.csv', index_col=0)
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)

    predictions_ball = one_hot_lstm(one_hot_ball_df, 5)
    predictions_star = one_hot_lstm(one_hot_star_df, 2)

    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}

if __name__=='__main__':
    
    euromillions = euromillions()
    # loto = loto()
    
    
    
    
    