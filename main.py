import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from modules.data_building import main as load_data
from modules.random_draw import random_draw

from modules.bilstm_draw import LotteryPredictorLSTM

ROOT_PATH = Path(__file__).parent

N_BALLS_EUROMILLIONS = 50 
K_BALL_DRAWN_EUROMILLIONS = 5
N_STARS_EUROMILLIONS = 12
K_STARS_DRAWN_EUROMILLIONS = 2

N_BALLS_LOTO = 49 
K_BALL_DRAWN_LOTO = 5
N_STARS_LOTO = 10
K_STARS_DRAWN_LOTO = 1

N_BALLS_EURODREAMS = 40 
K_BALL_DRAWN_EURODREAMS = 6
N_STARS_EURODREAMS = 5
K_STARS_DRAWN_EURODREAMS = 1


def random_predictions(lottery, seed=None):
    
    random_ball = random_draw(eval(f"N_BALLS_{lottery.upper()}"), eval(f"K_BALL_DRAWN_{lottery.upper()}"), seed)
    random_star = random_draw(eval(f"N_STARS_{lottery.upper()}"), eval(f"K_STARS_DRAWN_{lottery.upper()}"), seed)
    print(f'===================\nBall predictions: {random_ball}\n===================\nStar predictions: {random_star}\n===================\n')

    return random_ball, random_star

def bilstm_predictions(lottery):
    
    SEQUENCE_LENGTH = 30 
    EPOCHS = 50 
    
    all_one_hot_df = load_data(f'{lottery}')
    
    df_concat_ball_df = all_one_hot_df['ball_df']
    predictor = LotteryPredictorLSTM(
        sequence_length=SEQUENCE_LENGTH, 
        k_drawn=eval(f"K_BALL_DRAWN_{lottery.upper()}")
    )
    predictor.prepare_data(df_concat_ball_df)
    predictor.build_model()
    predictor.train_model(epochs=EPOCHS)
    last_sequence_ball = df_concat_ball_df.values[-SEQUENCE_LENGTH:]
    pred_ball = predictor.predict_next_draw(last_sequence_ball)
    
    df_concat_star_df = all_one_hot_df['star_df']
    predictor = LotteryPredictorLSTM(
        sequence_length=SEQUENCE_LENGTH, 
        k_drawn=eval(f"K_STARS_DRAWN_{lottery.upper()}")
    )
    predictor.prepare_data(df_concat_star_df)
    predictor.build_model()
    predictor.train_model(epochs=EPOCHS)
    last_sequence_star = df_concat_star_df.values[-SEQUENCE_LENGTH:]
    pred_star = predictor.predict_next_draw(last_sequence_star)
    
    print(f'===================\nBall predictions: {pred_ball["predictions"]}\nProbabilities: {np.round(pred_ball["top_k_probabilities"], 4)}\n===================\nStar predictions: {pred_star["predictions"]}\nProbabilities: {np.round(pred_star["top_k_probabilities"], 4)}\n===================\n')

if __name__=='__main__':

    LOTTERY = "eurodreams"
    STRATEGY = "bilstm"
    SEED = 0
    
    if STRATEGY == 'random':
        random_predictions(LOTTERY, SEED)
    elif STRATEGY == 'bilstm':
        bilstm_predictions(LOTTERY)