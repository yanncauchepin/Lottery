import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def meta_modeling(df, size, numbers):
    
    models = []
    y_pred_list = []
    y_probs_list = []

    
    # Iterate over each target variable (assuming columns represent target variables)
    for i in range(numbers):
        # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
    for i in range(numbers):
        pred.append(y_probs_list[i][0][0])
    next_y = np.argpartition(pred, -size)[-size:]
    next_y = [y+1 for y in next_y]
    pred = pd.DataFrame(pred)
    pred.index += 1
    pred = pred.sort_values(by=0, ascending=False)
    print(f'Next values: {next_y}')
    
    return pred, next_y

if __name__ == '__main__':
    
# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# 
# n_variables = 5
# n_times = 100
# indices = np.random.randint(0, n_variables, size=n_times)
# one_hot_time_series = np.eye(n_variables)[indices]
# 
# X = np.arange(n_times).reshape(-1, 1).astype(np.float64)  # GNB expects float64
# y = one_hot_time_series  # Use class labels directly
# 
# # Split data into train and test sets
# train_size = int(0.8 * n_times)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
# 
# # Initialize Gaussian Naive Bayes models for each output variable
# models = []
# for i in range(n_variables):
#     gnb = GaussianNB(priors=[1/n_variables]*n_variables)
#     gnb.fit(X_train, y_train[:, i])
#     models.append(gnb)
# 
# # Predict probabilities for each output variable
# y_probs = []
# for model in models:
#     probs = model.predict_proba(X_test)[:, 1]  # Get probabilities of the positive class
#     y_probs.append(probs)
# 
# # Convert list of probabilities to numpy array
# y_probs = np.array(y_probs).T
# 
# # Evaluate performance (Optional)
# # You can calculate accuracy if you have ground truth labels
# y_pred = np.argmax(y_probs, axis=1)  # Convert probabilities to class labels
# accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
# print("Accuracy:", accuracy)
# 
# # Print predicted probabilities
# print("Predicted Probabilities:", y_probs)
# =============================================================================

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    n_classes = 50
    n_times = 1000
    # Generate a unique variable of integers from 1 to 50
    labels = np.random.randint(1, n_classes + 1, size=n_times)
    
    X = np.arange(n_times).reshape(-1, 1).astype(np.float64)  # GNB expects float64
    y = labels  # Use integer labels directly
    
    # Split data into train and test sets
    train_size = int(0.8 * n_times)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize Gaussian Naive Bayes model
    gnb = GaussianNB(priors=[1/n_classes]*n_classes)
    
    # Fit the model
    gnb.fit(X_train, y_train)
    
    # Predict
    y_pred = gnb.predict(X_test)
    
    y_probs = gnb.predict_proba(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    print("Predicted Probabilities:", y_probs)
