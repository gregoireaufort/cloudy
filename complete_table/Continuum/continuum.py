# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:31:21 2020

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
import joblib
from matplotlib import pyplot as plt
 
os.environ['MdB_HOST']='3mdb.astro.unam.mx'
os.environ['MdB_USER']='OVN_user'
os.environ['MdB_PASSWD']='oiii5007'
os.environ['MdB_PORT']='3306'
os.environ['MdB_DBs']='3MdBs'
os.environ['MdB_DBp']='3MdB'
os.environ['MdB_DB_17']='3MdB_17'

sel = """SELECT
        com1 as com1,
        com2 as com2,
        com3 as com3,
        com4 as com4,
        com5 as com5,
        THp as THp,
        nH_mean as nH,
        POW(10, HELIUM)*A_HELIUM_vol_1/A_HYDROGEN_vol_1 as He1, 
        POW(10, HELIUM)*A_HELIUM_vol_2/A_HYDROGEN_vol_1 as He2,
        HbFrac
        FROM tab_17, abion_17 WHERE tab_17.N=abion_17.N """
db = pymysql.connect(host=os.environ['MdB_HOST'],
                                user=os.environ['MdB_USER'],
                                passwd=os.environ['MdB_PASSWD'],
                                db=os.environ['MdB_DB_17']) 
res = pd.read_sql(sel, con=db)
db.close()
N = res.shape[0]
a=np.array([res['com1'][i][0] =='l' for i in range(N)])
b=np.array([res['com2'][i][0] =='f' for i in range(N)])
c=np.array([res['com3'][i][0] =='a' for i in range(N)])
d=np.array([res['com4'][i][0] =='a' for i in range(N)])
e=np.array([res['com5'][i][0] =='N' for i in range(N)])
mask = a & b & c & d & e
params = ["com1", "com2", "com3", "com4", "com5","HbFrac"]
X= res[params][mask].reset_index(drop = True)

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

targets = ['He1', 'He2', 'THp', 'nH']

y= res[targets][mask]


scaler = StandardScaler()
X[params]= scaler.fit_transform(X=X[params])
scaler_y = StandardScaler()
y = scaler_y.fit_transform(X=y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=42)

dense = 512
model = MLPRegressor(max_iter=20000,
               tol=1e-7,
               solver='lbfgs',
               activation='tanh',
               hidden_layer_sizes=(dense, dense, dense,dense),
               random_state=1,
               verbose = True)
model.fit(X_train,y_train)

np.mean(np.square(model.predict(X_val)-y_val))


plt.scatter(model.predict(X_val)[:,0], y_val[:,0],s= 0.2)


import time
t1 = time.time()
model.predict(X[params])
print(time.time()-t1)

joblib.dump(model,"continuum/NN_512_cont.joblib")
joblib.dump(scaler, "continuum/scaler_continuum.joblib")
joblib.dump(scaler_y, "continuum/scaler_continuum_y.joblib")