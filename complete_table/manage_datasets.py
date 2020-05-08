# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:31:40 2020

@author: Gregoire Aufort
"""

import os
import pandas as pd
import pymysql
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def create_training_sets():
    try:
        res = pd.read_csv("data/Bond.csv")
    except : 
        co = pymysql.connect(host=os.environ['MdB_HOST'],
                                user=os.environ['MdB_USER'],
                                passwd=os.environ['MdB_PASSWD'],
                                db=os.environ['MdB_DB_17']) 
        res = pd.read_sql("select * from tab_17 where ref = 'BOND' ", con=co)
        co.close()
        res.to_csv("data/Bond.csv")
    
    
    lim_inf =np.where(res.columns=='ZINC')[0][0]
    lim_sup = np.where(res.columns=='MIPS_240000M')[0][0]
    
    tab = res.iloc[:,lim_inf:lim_sup]
    
    
    to_select = [(tab.columns[i].endswith('A') or tab.columns[i].endswith('M')) and not tab.columns[i].startswith('BLND')  for i in range(tab.shape[1])]
    raies = tab.columns[to_select]
    
    params = ["com1", "com2", "com3", "com4", "com5","HbFrac"]
    X= res[params]
    
    #need to retrieve the numbers in com1 --> com5
    valeur = pd.DataFrame(data= pd.np.empty(X.shape) * pd.np.nan,columns = params)
    for i in range(len(params)-1):
        print("param",params[i])
        for j in range(X.shape[0]):    
            for k in range(len(X[params[i]][j])):
                            if X[params[i]][j][k] =="=":
                                valeur[params[i]][j]=np.float(X[params[i]][j][(k+1):])
                                break
    valeur['HbFrac'] = X["HbFrac"]
    
    X =valeur
    
    X['com3'] /=1e6 #normalizing age 
    
    Hb = raies.get_loc("H__1_486133A")
    tab2=tab[raies].div(tab["H__1_486133A"],axis = 0)
    mask = tab2[tab2>0].quantile(q=0.5,axis =0)>1e-4
    y = np.log(tab2[raies[mask]])
    
    
    
    scaler = StandardScaler()
    X[params]= scaler.fit_transform(X=X[params])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=42)
    
    X_train.to_csv("data/X_train.csv")
    X_val.to_csv("data/X_val.csv")
    X_test.to_csv("data/X_test.csv")
    y_train.to_csv("data/y_train.csv")
    y_val.to_csv("data/y_val.csv")
    y_test.to_csv("data/y_test.csv")
    joblib.dump(scaler,"data/X_scaling.pkl")
    
def load_training_sets():
    X_train = pd.read_csv("data/X_train.csv",index_col=0)
    X_val = pd.read_csv("data/X_val.csv",index_col=0)
    y_train = pd.read_csv("data/y_train.csv",index_col=0)
    y_val = pd.read_csv("data/y_val.csv",index_col=0)
    
    return X_train,X_val,y_train,y_val

if __name__ == "__main__":
    create_training_sets()
    
