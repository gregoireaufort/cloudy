# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:35:23 2020

@author: Gregoire Aufort
"""


import os
import pandas as pd
import pymysql
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

sel = """SELECT
        H__1_486133A as Hb, 
        com1 as com1,
        com2 as com2,
        com3 as com3,
        com4 as com4,
        com5 as com5,
        HbFrac
        FROM tab_17
        WHERE ref = 'BOND' """
db = pymysql.connect(host=os.environ['MdB_HOST'],
                                user=os.environ['MdB_USER'],
                                passwd=os.environ['MdB_PASSWD'],
                                db=os.environ['MdB_DB_17']) 
res = pd.read_sql(sel, con=db)
db.close()

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

X =valeur.copy()

X['com3'] /=1e6 #normalizing age 

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = np.log(res["Hb"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=42)

clf = MLPRegressor(max_iter=20000,
                   tol=1e-8,
                   solver='lbfgs',
                   activation='tanh',
                   hidden_layer_sizes=(256, 256, 256),
                   random_state=1,
                   verbose = 2)


clf.fit(X_train,y_train)

clf.score(X_test,y_test)

pred = clf.predict(X_test)
print(np.mean(np.square(pred-y_test)))

print(np.mean(np.square(clf.predict(X_val)-y_val)))

print(np.mean((clf.predict(X_val)-y_val)/y_val))
print(np.mean(np.abs((pred-y_test)/y_test)))

joblib.dump(clf, "ANN/ANN_hbeta_256.pkl")