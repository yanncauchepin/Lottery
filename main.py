import numpy as np
import pandas as pd
from Lottery.modules.data_building import main as load_data
from Lottery.modules.model_lstm import meta_modeling as one_hot_lstm
from Lottery.modules.model_kalman_filter import meta_modeling as one_hot_kf
from Lottery.modules.model_particle_filter import meta_modeling as one_hot_pf
from Lottery.modules.model_transformers import meta_modeling as one_hot_transformers
from Lottery.modules.random_draw import random_draw


def loto_random():
    
    predictions_ball = random_draw(49, 5)
    predictions_star = random_draw(10, 1)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def euromillions_random():
    
    predictions_ball = random_draw(50, 5)
    predictions_star = random_draw(12, 2)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def eurodreams_random():
    
    predictions_ball = random_draw(40, 6)
    predictions_star = random_draw(5, 1)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def loto_one_hot_kf():

    load_data('loto', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_loto.csv', index_col=0)
    
    predictions_ball = one_hot_kf('loto_ball', one_hot_ball_df, 5, 49)
    predictions_star = one_hot_kf('loto_star', one_hot_star_df, 1, 10)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def euromillions_one_hot_kf():

    load_data('euromillions', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_euromillions.csv', index_col=0)
    
    predictions_ball = one_hot_kf('euromillions_ball', one_hot_ball_df, 5, 50)
    predictions_star = one_hot_kf('euromillions_star', one_hot_star_df, 2, 12)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def eurodreams_one_hot_kf():
    
    load_data('eurodreams', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_eurodreams.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_eurodreams.csv', index_col=0)

    predictions_ball = one_hot_kf('eurodreams_ball', one_hot_ball_df, 6, 40)
    predictions_star = one_hot_kf('eurodreams_star', one_hot_star_df, 1, 5)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def loto_one_hot_lstm():

    load_data('loto', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_loto.csv', index_col=0)
    
    predictions_ball = one_hot_lstm('loto_ball', one_hot_ball_df, 5, 49)
    predictions_star = one_hot_lstm('loto_star', one_hot_star_df, 1, 10)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def euromillions_one_hot_lstm():

    load_data('euromillions', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_euromillions.csv', index_col=0)
    
    predictions_ball = one_hot_lstm('euromillions_ball', one_hot_ball_df, 5, 50)
    predictions_star = one_hot_lstm('euromillions_star', one_hot_star_df, 2, 12)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def eurodreams_one_hot_lstm():
    
    load_data('eurodreams', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_eurodreams.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_eurodreams.csv', index_col=0)

    predictions_ball = one_hot_lstm('eurodreams_ball', one_hot_ball_df, 6, 40)
    predictions_star = one_hot_lstm('eurodreams_star', one_hot_star_df, 1, 5)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def loto_one_hot_particles_filter():

    load_data('loto', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_loto.csv', index_col=0)
    
    predictions_ball = one_hot_pf('loto_ball', one_hot_ball_df, 5, 49)
    predictions_star = one_hot_pf('loto_star', one_hot_star_df, 1, 10)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def euromillions_one_hot_particles_filter():

    load_data('euromillions', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_euromillions.csv', index_col=0)
    
    predictions_ball = one_hot_pf('euromillions_ball', one_hot_ball_df, 5, 50)
    predictions_star = one_hot_pf('euromillions_star', one_hot_star_df, 2, 12)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def eurodreams_one_hot_particles_fitler():
    
    load_data('eurodreams', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_eurodreams.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_eurodreams.csv', index_col=0)

    predictions_ball = one_hot_pf('eurodreams_ball', one_hot_ball_df, 6, 40)
    predictions_star = one_hot_pf('eurodreams_star', one_hot_star_df, 1, 5)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def loto_one_hot_particle_transformers():

    load_data('loto', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_loto.csv', index_col=0)
    
    predictions_ball = one_hot_transformers('loto_ball', one_hot_ball_df, 5, 49)
    predictions_star = one_hot_transformers('loto_star', one_hot_star_df, 1, 10)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def euromillions_one_hot_transformers():

    load_data('euromillions', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_euromillions.csv', index_col=0)
    
    predictions_ball = one_hot_transformers('euromillions_ball', one_hot_ball_df, 5, 50)
    predictions_star = one_hot_transformers('euromillions_star', one_hot_star_df, 2, 12)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

def eurodreams_one_hot_transformers():
    
    load_data('eurodreams', 'concatenate_one_hot')
    one_hot_ball_df = pd.read_csv('data/all_concat_one_hot_ball_eurodreams.csv', index_col=0)
    one_hot_star_df = pd.read_csv('data/all_concat_one_hot_star_eurodreams.csv', index_col=0)

    predictions_ball = one_hot_transformers('eurodreams_ball', one_hot_ball_df, 6, 40)
    predictions_star = one_hot_transformers('eurodreams_star', one_hot_star_df, 1, 5)
    print(f'===================\nBall predictions: {predictions_ball}\n===================\nStar predictions: {predictions_star}\n===================\n')

    return predictions_ball, predictions_star

if __name__=='__main__':
    pass

    # euromillions_random = euromillions_random()
    # loto_random = loto_random()
    # eurodreams_random = eurodreams_random()

    # predictions_ball, predictions_star = loto_one_hot_kf()
    # predictions_ball, predictions_star = euromillions_one_hot_kf()
    # predictions_ball, predictions_star = eurodreams_one_hot_kf()

    # predictions_ball, predictions_star = loto_one_hot_pf()
    # predictions_ball, predictions_star = euromillions_one_hot_pf()
    # predictions_ball, predictions_star = eurodreams_one_hot_pf()

    # predictions_ball, predictions_star = loto_one_hot_lstm()
    # predictions_ball, predictions_star = euromillions_one_hot_lstm()
    # predictions_ball, predictions_star = eurodreams_one_hot_lstm()

    # predictions_ball, predictions_star = loto_one_hot_transformers()
    # predictions_ball, predictions_star = euromillions_one_hot_transformers()
    # predictions_ball, predictions_star = eurodreams_one_hot_transformers()