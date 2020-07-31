# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:44:25 2020

@author: Kyle
"""
import sudml
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor
import os
from itertools import product





J = 100
mse1 = np.zeros(100)
mse2 = np.zeros(100)

for i in range(J):
    data = sudml.DGP1(1000,0)
    data = (data - data.mean()) / (data.max() - data.min())
    if data.min().any()<0:
        data = data - data.min()
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    
    train_small = train.sample(n=100, random_state=1)
    model1 = sudml.NeuralNet1(101)
    model1.fit(np.array(train.drop(['Y1','Y2'],axis=1)),np.array(train['Y1']))
    j=0
    for param in model1.parameters():
        if j<4:
            param.requires_grad = False
            j+=1
    model1.fit(np.array(train_small.drop(['Y1','Y2'],axis=1)),np.array(train_small['Y2']))
    pred1 = model1.predict(np.array(test.drop(['Y1','Y2'],axis=1)))
    mse1[i] = np.mean((pred1-np.array(test['Y2']))**2)



    model2 = sudml.NeuralNet1(101)
    model2.fit(np.array(train_small.drop(['Y1','Y2'],axis=1)),np.array(train_small['Y2']))
    pred2 = model2.predict(np.array(test.drop(['Y1','Y2'],axis=1)))
    mse2[i] = np.mean((pred2-np.array(test['Y2']))**2)

    print(mse1[i])
    print(mse2[i])

print(np.mean(mse1))
print(np.mean(mse2))



for param in model.parameters():
    print(param)








print(pred2[0:10])
print(train['Y1'][0:10])

for i in range(J):
    print(i)
    model = sudml.MVNeuralNet1(101)
    model.fit(np.array(train.drop(['Y1','Y2'],axis=1)),np.array(train[['Y1','Y2']]))
    pred1 = model.predict(np.array(train.drop(['Y1','Y2'],axis=1)))
    mse1[i] = np.mean((pred1[:,0]-np.array(train['Y1']))**2)
    #print(mse1[i])
    
    model = sudml.NeuralNet1(101)
    model.fit(np.array(train.drop(['Y1','Y2'],axis=1)),np.array(train['Y1']))
    pred2 = model.predict(np.array(train.drop(['Y1','Y2'],axis=1)))
    mse2[i] = np.mean((pred2-np.array(train['Y1']))**2)

print(np.nanmean(mse1))
print(np.nanmean(mse2))


import matplotlib.pyplot as plt
x = [0,0.2,0.4,0.6,0.8,1]
y1 = [1,1,1,1,1,1]
y2 = [0.88,0.90,0.93,0.96,0.97,.99]
plt.figure(figsize=(4,4))
plt.plot(x, y1, 'b-',label="Baseline")
plt.plot(x, y2, 'g-',label="Multi-task Learning")
plt.title('Efficiency of Multi-task Learning')
plt.ylabel('Relative MSE')
plt.xlabel('Heterogeneity')
plt.ylim(0,1.5)
plt.tight_layout()
plt.legend()
plt.show()


x = [0,0.2,0.4,0.6,0.8,1]
y1 = [1,1,1,1,1,1]
y2 = [0.72,0.82,0.86,0.89,0.97,.99]
plt.figure(figsize=(4,4))
plt.plot(x, y1, 'b-',label="Baseline")
plt.plot(x, y2, 'g-',label="Transfer Learning")
plt.title('Efficiency of Transfer Learning')
plt.ylabel('Relative MSE')
plt.xlabel('Heterogeneity')
plt.ylim(0,1.5)
plt.tight_layout()
plt.legend()
plt.show()





plt.savefig(f)




















args_lasso1 = {
        'alpha':0.00418519,
        'max_iter':5000,
        'normalize':True,
        'tol':0.001
        }

args_lasso2 = {
        'alpha':0.00281957,
        'max_iter':5000,
        'normalize':True,
        'tol':0.001
        }



model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)


L=5
# X = data.drop(['Y1','Y2','T'],axis=1)
# T = data['T']
# Y = data['Y1']
# model = sudml.dmlATE(model_lasso1,model_lasso2)
# model.fit(data.drop(['Y1','Y2','T'],axis=1),data['T'],data['Y1'],L=L,basis=False,standardize=False)


for i in range(100):
    data = sudml.DGP1(1000)
    model = sudml.dmlATE(model_lasso1,model_lasso2)
    model.fit(data.drop(['Y1','Y2','T'],axis=1),data['T'],data['Y1'],L=L,basis=False,standardize=False)
    print(model.ATE)



















