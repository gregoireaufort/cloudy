# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:03:32 2020

@author: groet
"""



import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import normalize, MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from sklearn.externals import joblib

np.random.seed(42)


def create_model(dense_1,dropout_rate = 0.1,lr=0.05):
    # create model
    model = Sequential()
    input_shape =(6,)
    model.add(Dense(dense_1, 
                    input_shape =input_shape, 
                    activation="relu"))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(5, activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', 
                  optimizer='adam', 
                  metrics=['mean_squared_error'])
    return model

    
co = pymysql.connect(host='3mdb.astro.unam.mx', db='3MdB_17', user='OVN_user', passwd='oiii5007')
res = pd.read_sql("select com1, com2, com3, com4, com5,HbFrac, N__2_654805A, N__2_658345A,  H__1_656281A, H__1_486133A, O__3_500684A from tab_17 where ref = 'BOND' and HbFrac > 0.7", con=co)
co.close()

#res.to_csv('raies_3mdb.csv', sep='\t')

#, "CA_B_656281A", "CA_B_486133A",
res.head()
raies = ["N__2_654805A", "N__2_658345A",  "H__1_656281A", "H__1_486133A", "O__3_500684A"]
y =  np.array(np.log(res[raies]))
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
X = np.array(valeur)
X_max,X_min = np.max(X, axis = 0),np.min(X,axis = 0)
y_maxy_min = np.max(y, axis = 0),np.min(y,axis = 0)
scaler = MinMaxScaler()
X = scaler.fit_transform(X=X)
#y = scaler.fit_transform(X=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KerasRegressor(build_fn=create_model, verbose=0)

##############################################################
# grid search
lr = [0.001, 0.01,0.1]
dropout_rate = [0.1,0.2]
dense_1 = [32,64,128,256]
batch_size = [1000] 
epochs = [1000]
param_grid = dict(epochs=epochs, 
                  batch_size=batch_size, 
                  lr=lr,
                  dropout_rate = dropout_rate, 
                  dense_1 = dense_1)

kfold_splits =5
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,return_train_score=True,
                cv=kfold_splits, verbose =0)

#grid_result = grid.fit(X_train, y_train) 

##############################################################

#joblib.dump(grid.best_estimator_, 'NN_lines.pkl', compress = 1)

model = joblib.load('NN_lines.pkl')

y_est_test = model.predict(X_test)

y_true = y_test

print(np.mean(np.square(y_true - y_est_test)))
print(np.max(y_true), np.max(y_est_test), np.max(np.mean((np.abs(y_true - y_est_test)/y_true),axis=0)))



plt.scatter(y_true[:,0],y_est_test[:,0])
plt.scatter(y_true[:,1],y_est_test[:,1])

plt.scatter(y_true[:,2],y_est_test[:,2])



renorm_est = np.exp(np.float64(y_est_test))
renorm_true = np.exp(y_true)

for i in range(5):
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(renorm_true[:,i],renorm_est[:,i] , c='blue', alpha=0.1, edgecolors='none')
    ax.set_title(raies[i])
    plt.xlabel('True', fontsize=18)
    plt.ylabel('Estimated', fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig(raies[i]+'.png')


fig = plt.figure()
ax = plt.gca()
relative = np.mean((renorm_true - renorm_est)/renorm_true,axis = 1)
relative_trunc = relative[np.abs(relative) <20]
plt.plot(relative_trunc,"+")



bpt_true = pd.DataFrame(data = {"O3/Hb": np.log(renorm_true[:,4]/renorm_true[:,3]),
                        "N2/Ha" : np.log(renorm_true[:,1]/renorm_true[:,2])})



bpt_est = pd.DataFrame(data = {"O3/Hb": np.log(renorm_est[:,4]/renorm_est[:,3]),
                        "N2/Ha" : np.log(renorm_est[:,1]/renorm_est[:,2])})


plt1 = plt.scatter(bpt_true["O3/Hb"],bpt_true["N2/Ha"],alpha=0.3, label = "true")
plt2 = plt.scatter(bpt_est["O3/Hb"],bpt_est["N2/Ha"], color = 'red',alpha=0.3,label = "estimated")
plt.xlabel("O3/Hb", fontsize=18)
plt.ylabel("N2/Ha", fontsize=16)
plt.legend(handles=[plt1, plt2])
#plt.savefig('BPT.png')
plt.show()



t = np.linspace(-20,0.4, 400)
tr = (0.61/(t-0.47))+1.19
t2 = np.linspace(-20,0.05, 400)
tr2 = (0.61/(t2-0.05))+1.3
t3 = np.linspace(-20,0.1, 400)
tr3 = (0.61/(t3-0.02-(0.1833*0.8)))+1.2+(0.03*0.8)
t4 = np.linspace(-20,0.27, 400)
tr4 = (0.61/(t4-0.02-(0.1833*1.5)))+1.2+(0.03*1.5)
t5 = np.linspace(-20,0.45, 400)
tr5 = (0.61/(t5-0.02-(0.1833*2.5)))+1.2+(0.03*2.5)

plt.plot(t3,tr3)


ax = plt.subplot(1,1,1)
plt1 = ax.scatter(bpt_true["N2/Ha"],bpt_true["O3/Hb"],alpha=0.3, label = "true")
plt2 = ax.scatter(bpt_est["N2/Ha"],bpt_est["O3/Hb"], color = 'red',alpha=0.3,label = "estimated")
plt3 = ax.plot(t,tr, label = "Kewley +01", color ="green")
plt4 = ax.plot(t2,tr2, label = "Kauffman +03")
plt5 =ax.plot(t3,tr3, label = "Kewley +13, z=0.8")
plt6 = ax.plot(t4,tr4, label = "Kewley +13, z=1.5")
plt7 = ax.plot(t5,tr5, label = "Kewley +13, z=2.5")
plt.xlabel("LOG(NII/Ha)", fontsize=18)
plt.ylabel("LOG(OIII/Hb)", fontsize=16)
plt.xlim(-2,2)
plt.ylim(-2,2)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.savefig('BPT_zoom_lines.png')
plt.show()

ax = plt.subplot(1,1,1)
plt1 = ax.scatter(bpt_true["N2/Ha"],bpt_true["O3/Hb"],alpha=0.3, label = "true")
plt2 = ax.scatter(bpt_est["N2/Ha"],bpt_est["O3/Hb"], color = 'red',alpha=0.3,label = "estimated")
plt3 = ax.plot(t,tr, label = "Kewley +01", color ="green")
plt4 = ax.plot(t2,tr2, label = "Kauffman +03")
plt5 =ax.plot(t3,tr3, label = "Kewley +13, z=0.8")
plt6 = ax.plot(t4,tr4, label = "Kewley +13, z=1.5")
plt7 = ax.plot(t5,tr5, label = "Kewley +13, z=2.5")
plt.xlabel("LOG(NII/Ha)", fontsize=18)
plt.ylabel("LOG(OIII/Hb)", fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.savefig('BPTlines.png')
plt.show()

verif1 =np.log( res["N__2_658345A"]/res["H__1_656281A"])
verif2 =np.log( res["O__3_500684A"] / res["H__1_486133A"])
plt.scatter(verif1,verif2)
plt.plot(t2,tr2, label = "Kauffman +03",color = "red")
