import numpy as np
import pandas as pd
from pathlib import Path
from modules.data_building import main as load_data
from modules.random_draw import random_draw

ROOT_PATH = Path(__file__).parent

def loto_random(seed=None):
    
    random_ball = random_draw(49, 5, seed)
    random_star = random_draw(10, 1, seed)
    print(f'===================\nBall predictions: {random_ball}\n===================\nStar predictions: {random_star}\n===================\n')

    return random_ball, random_star

def euromillions_random(seed=None):
    
    random_ball = random_draw(50, 5, seed)
    random_star = random_draw(12, 2, seed)
    print(f'===================\nBall predictions: {random_ball}\n===================\nStar predictions: {random_star}\n===================\n')

    return random_ball, random_star

def eurodreams_random(seed=None):
    
    random_ball = random_draw(40, 6, seed)
    random_star = random_draw(5, 1, seed)
    print(f'===================\nBall predictions: {random_ball}\n===================\nStar predictions: {random_star}\n===================\n')

    return random_ball, random_star

if __name__=='__main__':

    # euromillions_random(seed=20251027)
    # euromillions_random(seed=20251102)
    
    # loto_random(seed=20251027)
    # loto_random(seed=20251102)

    # eurodreams_random(seed=20251027)
    # eurodreams_random(seed=20251102)
