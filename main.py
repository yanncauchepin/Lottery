import numpy as np
import pandas as pd
from Lottery.euromillions.data_building import main as euromillions_data
from Lottery.loto.data_building import main as loto_data
from Lottery.model.one_hot_lstm import meta_modeling as one_hot_lstm
from Lottery.model.one_hot_gnb import meta_modeling as one_hot_gnb
from Lottery.model.random_draw import random_draw


def loto_gnb():
    
    loto_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_loto.csv', index_col=0)
    
    predictions_ball = one_hot_gnb(one_hot_ball_df, 5, 49)
    predictions_star = one_hot_gnb(one_hot_star_df, 1, 10)
    
    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


def loto_lstm():
    
    loto_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_loto.csv', index_col=0)

    predictions_ball = one_hot_lstm(one_hot_ball_df, 5)
    predictions_star = one_hot_lstm(one_hot_star_df, 1)

    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


def loto_random():
    
    predictions_ball = random_draw(49, 5)
    predictions_star = random_draw(10, 1)
    
    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


def euromillions_gnb():
    
    euromillions_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)
    
    predictions_ball = one_hot_gnb(one_hot_ball_df, 5, 50)
    predictions_star = one_hot_gnb(one_hot_star_df, 2, 12)
    
    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}

def euromillions_lstm():
    
    euromillions_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)

    predictions_ball = one_hot_lstm(one_hot_ball_df, 5)
    predictions_star = one_hot_lstm(one_hot_star_df, 2)

    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


def euromillions_random():
    
    predictions_ball = random_draw(50, 5)
    predictions_star = random_draw(12, 2)
    
    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


def eurodreams_random():
    
    predictions_ball = random_draw(40, 6)
    predictions_star = random_draw(5, 1)
    
    return {'predictions_ball': predictions_ball, 'predictions_star': predictions_star}


if __name__=='__main__':
    
    euromillions_gnb = euromillions_gnb()
    euromillions_lstm = euromillions_lstm()
    euromillions_random = euromillions_random()
    # eurodreams_random = eurodreams_random()
    # loto_gnb = loto_gnb()
    # loto_lstm = loto_lstm()
    # loto_random = loto_random()
    
    
# =============================================================================
#     EXPERIMENTAL EUROMILLIONS
    sum_prob_ball = euromillions_gnb["predictions_ball"][0].add(
        euromillions_lstm["predictions_ball"][1])
    sum_prob_ball = sum_prob_ball.sort_values(by=0, ascending=False)
    sum_prob_star = euromillions_gnb["predictions_star"][0].add(
        euromillions_lstm["predictions_star"][1])
    sum_prob_star = sum_prob_star.sort_values(by=0, ascending=False)

# =============================================================================

# =============================================================================
#     EXPERIMENTAL LOTO
    # sum_prob_ball = loto_gnb["predictions_ball"][0].add(
    #     loto_lstm["predictions_ball"][1])
    # sum_prob_ball = sum_prob_ball.sort_values(by=0, ascending=False)
    # sum_prob_star = loto_gnb["predictions_star"][0].add(
    #     loto_lstm["predictions_star"][1])
    # sum_prob_star = sum_prob_star.sort_values(by=0, ascending=False)
# =============================================================================
  
    