# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:11:40 2020

@author: Gregoire Aufort
"""

from manage_datasets import *
from utils import pca
from sklearn.multioutput import MultiOutputRegressor
import time
from GPyOpt.methods import BayesianOptimization
from lightgbm import LGBMRegressor


def get_model(learning_rate,  n_estimators, max_depth, min_data_in_leaf,num_leaves):
    model =  MultiOutputRegressor(
        LGBMRegressor(learning_rate = learning_rate,
                         n_estimators = n_estimators,
                         max_depth = max_depth,
                         min_data_in_leaf = min_data_in_leaf,
                         num_leaves = num_leaves,
                         subsample = 0.7,
                         objective = 'regression',
                         n_jobs = 24,
                         random_state=42))
    return model

def objective(li):

    #Create the model using a specified hyperparameters.
    parameters = li[0]
    learning_rate=10**(-parameters[0])
    min_data_in_leaf=int(parameters[1])
    max_depth=int(parameters[2])
    n_estimators=int(parameters[3])
    num_leaves = int(parameters[4]*(2**max_depth))
    model = get_model(learning_rate=learning_rate,
                              min_data_in_leaf=min_data_in_leaf,
                              max_depth=max_depth,
                              n_estimators=n_estimators,
                              num_leaves = num_leaves)   
    model.fit(X=X1,y=y1)
    # Evaluate the model with the eval dataset.
    score = np.mean(np.square(model.predict(X2)-y2))
    namefile = 'LGBM/'+time.strftime("%d_%H_%M_%S")+'.pkl'
    joblib.dump(model,namefile)
    return score

def gp_opt():
    bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (1, 5)},
        {'name': 'min_data_in_leaf', 'type': 'continuous', 'domain': (30, 300)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (5, 17)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (500, 3000)},
        {'name': 'num_leaves', 'type': 'continuous', 'domain': (0.3, 1)}]

    optimizer = BayesianOptimization(f=objective, 
                                  domain=bds,
                                  model_type='GP',
                                  acquisition_type ='EI',
                                  acquisition_jitter = 0.05,
                                  exact_feval=True, 
                                  maximize=False,
                                  num_cores = 24)
    optimizer.run_optimization(max_iter=100)
    joblib.dump(optimizer,"LGBM/gp_optimizer.pkl")
    

def main():
    global X1, X2, y1, y2 
    X1, X2, y1, y2 = load_training_sets()
    y1, y2 = pca(y1,y2)
    gp_opt()

if __name__ == "__main__":
    main()
    
    