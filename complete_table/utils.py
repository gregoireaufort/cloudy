# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:08:29 2020

@author: Gregoire Aufort
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.externals import joblib



def pca(y_train, y_val):
    Pca = PCA(n_components=0.9995, whiten = True)
    Pca.fit(y_train)
    joblib.dump(Pca,"pca_trained.pkl")
    return Pca.transform(y_train), Pca.transform(y_val)