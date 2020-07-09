# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:40:12 2020

@author: Gregoire Aufort
"""



import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import joblib

import pyneb as pn

class deep_continuum(object):
    def __init__(self):
        self.model = joblib.load("continuum/NN_512_cont.joblib")
        self.scalerx = joblib.load("continuum/scaler_continuum.joblib")
        self.scalery = joblib.load("continuum/scaler_continuum_y.joblib")
        self.pyneb_continuum = pn.Continuum()
        
    def compute_continuum(self,logU,geom_factor,age,Log_O_H,log_N_O, HbFrac,wl):
        n = 1
        model_inputs= self.scalerx.inverse_transform(np.array([logU,
                                                               geom_factor,
                                                               age,
                                                               Log_O_H,
                                                               log_N_O,
                                                               HbFrac]))
        pred_params_pyneb = self.model.predict(model_inputs)
        params_pyneb = self.scalery.inverse_transform( pred_params_pyneb)
        continuum = [self.pyneb_continuum.get_continuum(tem = params_pyneb[0],
                                                       den= params_pyneb[1],
                                                       He1_H = params_pyneb[2],
                                                       He2_H = params_pyneb[3],
                                                       wl =wl) for i in range(n)]
        
        return continuum
    