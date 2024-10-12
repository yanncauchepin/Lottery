import numpy as np
from datetime import datetime

def random_draw (max_balls, number_draws):
    seed = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed)
    values = np.random.permutation(np.arange(1, max_balls + 1))[:number_draws].tolist()
    print(f'Next : {values}')
    return values
    