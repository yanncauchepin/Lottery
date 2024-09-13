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
    
    pred_next_ball, next_ball = one_hot_gnb(one_hot_ball_df, 5, 49)
    pred_next_star, next_star = one_hot_gnb(one_hot_star_df, 1, 10)
    
    return {
        'pred_next_ball': pred_next_ball,
        'next_ball': next_ball,
        'pred_next_star': pred_next_star,
        'next_star': next_star
        }


def loto_lstm():
    
    loto_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_loto.csv', index_col=0)

    _, pred_next_ball, next_ball = one_hot_lstm(one_hot_ball_df, 5)
    _, pred_next_star, next_star = one_hot_lstm(one_hot_star_df, 1)

    return {
        'pred_next_ball': pred_next_ball,
        'next_ball': next_ball,
        'pred_next_star': pred_next_star,
        'next_star': next_star
        }

def loto_random():
    
    predictions_ball = random_draw(49, 5)
    predictions_star = random_draw(10, 1)
    
    return {
        'next_ball': next_ball,
        'next_star': next_star
        }

def loto_ensemble():
    
    lstm = loto_lstm()
    gnb = loto_gnb()
    
    sum_pred_ball = gnb["pred_next_ball"].add(
        lstm["pred_next_ball"])
    sum_pred_ball = sum_pred_ball.sort_values(by=0, ascending=False)
    print(f"Ensemble ball:\n{sum_pred_ball}")
    
    sum_pred_star = gnb["pred_next_star"].add(
        lstm["pred_next_star"])
    sum_pred_star = sum_pred_star.sort_values(by=0, ascending=False)
    print(f"Ensemble star:\n{sum_pred_star}")
    
    return {
        'sum_pred_ball': sum_pred_ball,
        'sum_pred_star': sum_pred_star
    }

def euromillions_gnb():
    
    euromillions_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)
    
    pred_next_ball, next_ball = one_hot_gnb(one_hot_ball_df, 5, 50)
    pred_next_star, next_star = one_hot_gnb(one_hot_star_df, 2, 12)
    
    return {
        'pred_next_ball': pred_next_ball,
        'next_ball': next_ball,
        'pred_next_star': pred_next_star,
        'next_star': next_star
        }
    
def euromillions_lstm():
    
    euromillions_data()
    
    one_hot_ball_df = pd.read_csv('data/all_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_one_hot_star_euromillions.csv', index_col=0)

    _, pred_next_ball, next_ball = one_hot_lstm(one_hot_ball_df, 5)
    _, pred_next_star, next_star = one_hot_lstm(one_hot_star_df, 2)

    return {
        'pred_next_ball': pred_next_ball,
        'next_ball': next_ball,
        'pred_next_star': pred_next_star,
        'next_star': next_star
        }

def euromillions_random():
    
    predictions_ball = random_draw(50, 5)
    predictions_star = random_draw(12, 2)
    
    return {
        'next_ball': next_ball,
        'next_star': next_star
        }

def euromillions_ensemble():
    
    lstm = euromillions_lstm()
    gnb = euromillions_gnb()
    
    sum_pred_ball = gnb["pred_next_ball"].add(
        lstm["pred_next_ball"])
    sum_pred_ball = sum_pred_ball.sort_values(by=0, ascending=False)
    print(f"Ensemble ball:\n{sum_pred_ball}")
    
    sum_pred_star = gnb["pred_next_star"].add(
        lstm["pred_next_star"])
    sum_pred_star = sum_pred_star.sort_values(by=0, ascending=False)
    print(f"Ensemble star:\n{sum_pred_star}")
    
    return {
        'sum_pred_ball': sum_pred_ball,
        'sum_pred_star': sum_pred_star
    }

def eurodreams_random():
    
    predictions_ball = random_draw(40, 6)
    predictions_star = random_draw(5, 1)
    
    return {
        'next_ball': next_ball,
        'next_star': next_star
        }

if __name__=='__main__':
    
    pass
    
    # euromillions_gnb = euromillions_gnb()
    # euromillions_lstm = euromillions_lstm()
    # euromillions_random = euromillions_random()
    # euromillions_ensemble = euromillions_ensemble()
    
    # loto_gnb = loto_gnb()
    # loto_lstm = loto_lstm()
    # loto_random = loto_random()
    loto_ensemble = loto_ensemble()
    
    # eurodreams_random = eurodreams_random()