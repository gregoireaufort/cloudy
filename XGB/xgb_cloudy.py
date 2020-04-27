import os
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
np.random.seed(42)

   
co = pymysql.connect(host='3mdb.astro.unam.mx', db='3MdB_17', user='OVN_user', passwd=os.environ['3mdb_pwd'])
res = pd.read_sql("select com1, com2, com3, com4, com5,HbFrac, N__2_654805A, N__2_658345A,  H__1_656281A, H__1_486133A, O__3_500684A, O__1_630030A,S__2_671644A   from tab_17 where ref = 'BOND' and HbFrac > 0.7", con=co)
co.close()

#res.to_csv('raies_3mdb.csv', sep='\t')

#, "CA_B_656281A", "CA_B_486133A",
res.head()
raies = ["N__2_654805A", "N__2_658345A",  "H__1_656281A", "H__1_486133A", "O__3_500684A", "O__1_630030A","S__2_671644A"]
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


xgb_model =  MultiOutputRegressor(xgb.XGBRegressor(objective = 'reg:squarederror',n_jobs = 14))
clf = GridSearchCV(xgb_model,
                   {'estimator__max_depth': [2,6,10,12],
                    'estimator__n_estimators': [100,200,500]},
                   verbose=0,
                   cv = 10,
                   scoring="neg_mean_squared_error")
clf.fit(X_train,y_train)
print(clf.best_score_)
print(clf.best_params_)


joblib.dump(clf.best_estimator_, 'XGB_lines.pkl', compress = 1)

model = joblib.load('XGB_lines.pkl')

y_est_test = model.predict(X_test)
print(np.mean(y_est_test-y_test)**2)
plt.scatter(y_est_test[:,6],y_test[:,6])




y_true = y_test

print(np.mean(np.square(y_true - y_est_test)))
print(np.max(y_true), np.max(y_est_test), np.max(np.mean((np.abs(y_true - y_est_test)/y_true),axis=0)))



plt.scatter(y_true[:,0],y_est_test[:,0])
plt.scatter(y_true[:,1],y_est_test[:,1])

plt.scatter(y_true[:,2],y_est_test[:,2])



renorm_est = np.exp(np.float64(y_est_test))
renorm_true = np.exp(y_true)

for i in range(7):
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(renorm_true[:,i],renorm_est[:,i] , c='blue', alpha=0.5, edgecolors='none', s= 7)
    ax.set_title(raies[i])
    plt.xlabel('True', fontsize=18)
    plt.ylabel('Estimated', fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig(raies[i]+'_7raies_xgb.png')


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
plt.savefig('BPT_zoom_lines_XGB.png')
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
plt.savefig('BPTlines_XGB.png')
plt.show()

