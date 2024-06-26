import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def meta_modeling(df, size, numbers):
    
    models = []
    y_pred_list = []
    y_probs_list = []

    
    # Iterate over each target variable (assuming columns represent target variables)
    for i in range(numbers):

        # Extract the target variable
        X = np.arange(len(df)).reshape(-1, 1).astype(np.int32)
        y = df.iloc[:, i].values.ravel()

        # Train a Gaussian Naive Bayes model for the current target variable
        gnb = GaussianNB(priors=[1/numbers, 1-1/numbers])
        gnb.fit(X, y)
        models.append(gnb)

        # Predict the next values for the current target variable
        last_X = np.arange(df.shape[0] + size)[-size:].reshape(-1, 1).astype(np.int32)
        y_pred = gnb.predict(last_X)
        y_probs = gnb.predict_proba(last_X)
        y_pred_list.append(y_pred)
        y_probs_list.append(y_probs)
    
    pred = list()

    for i in range(numbers):
        pred.append(y_probs_list[i][0][0])
    next_y = np.argpartition(pred, -size)[-size:]
    next_y = [y+1 for y in next_y]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Next values: {next_y}')
    
    return pred, next_y
