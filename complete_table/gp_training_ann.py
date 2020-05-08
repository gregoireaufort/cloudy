# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:03:17 2020

@author: Gregoire Aufort
"""


import numpy as np
from sklearn.externals import joblib
import time
from skopt import gp_minimize, dump
from skopt.space import Real, Integer
from sklearn.neural_network import MLPRegressor
from manage_datasets import *
from utils import pca


def objective(li):

    # Create the model using a specified hyperparameters.
    dense1 = max(2**int(li[0]),8)
    dense2 =max(2**int(li[1]),8)
    dense3 = max(2**int(li[2]),8)
    dense4 = max(2**int(li[3]),8)
    model = MLPRegressor(max_iter=20000,
                   tol=1e-7,
                   solver='lbfgs',
                   activation='tanh',
                   hidden_layer_sizes=(dense1, dense2, dense3,dense4),
                   random_state=1,
                   verbose = 1)
    
    t1 = time.time()
    model.fit(X=X1,y=y1)
    score =np.mean(np.square(model.predict(X2)-y2))
    print(time.time()-t1)
    namefile = 'ANN/'+time.strftime("%d_%H_%M_%S")+'.pkl'
    joblib.dump(model,namefile)
    return score

def gp_opt():
    space  = [Integer(3,8 , name='dense1'),
              Integer(3, 8, name='dense2'),
              Integer(3, 8, name='dense3'),
              Integer(3, 8, name='dense4')]
    
    res_gp = gp_minimize(objective, space, n_calls=25, random_state=0)
    dump(res_gp,"ANN/gp_optimized_25_pca.pkl")

def main():
    global X1, X2, y1, y2 
    X1, X2, y1, y2 = load_training_sets()
    y1, y2 = pca(y1,y2)
    gp_opt()

if __name__ == "__main__":
    main()
    
    
    