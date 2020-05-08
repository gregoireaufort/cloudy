# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:11:40 2020

@author: Gregoire Aufort
"""

from manage_datasets import *
from utils import pca
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from skopt.space import Real, Integer
from skopt import gp_minimize, dump
import time
from GPyOpt.methods import BayesianOptimization



def get_model(learning_rate,  n_estimators, max_depth, gamma):
    model =  MultiOutputRegressor(
        xgb.XGBRegressor(learning_rate = learning_rate,
                         n_estimators = n_estimators,
                         max_depth = max_depth,
                         gamma = gamma,
                         subsample = 0.7,
                         objective = 'reg:squarederror',
                         n_jobs = 16,
                         random_state=42,
                         tree_method = 'hist'))
    return model

def objective(li):

    # Create the model using a specified hyperparameters.
    max_depth = max(int(li[0]),1)
    gamma =li[3]
    learning_rate = li[1]
    n_estimators =  max(int(li[2])*200,200)

    t1 = time.time()
    model.fit(X=X1,y=y1)
    # Evaluate the model with the eval dataset.
    score = np.mean(np.square(model.predict(X2)-y2)) #put a minus if using bayes_opt
    print(time.time()-t1)
    namefile = 'XGB/'+time.strftime("%d_%H_%M_%S")+'.pkl'
    joblib.dump(model,namefile)
    return score

def gp_opt():
    space  = [Integer(7, 12, name='max_depth'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
          Integer(5, 10, name='n_estimators'),
          Real(0,30,"uniform", name = "gamma")]

    optimizer= gp_minimize(fit_with, space, n_calls=25, random_state=0)
    dump(optimizer,"XGB/gp_optimized_25_pca.pkl")
    

def main():
    global X1, X2, y1, y2 
    X1, X2, y1, y2 = load_training_sets()
    y1, y2 = pca(y1,y2)
    gp_opt()

if __name__ == "__main__":
    main()
    
    