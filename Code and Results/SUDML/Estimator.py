# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:45:09 2020

@author: Kyle
"""
import numpy as np
from scipy.stats import norm
import pandas as pd
import copy
import sklearn

class dmlATE:
    def __init__(self,model1,model2):
        self.ATE = None
        self.std_errors = None
        self.model1 = model1
        self.model2 = model2
        self.summary = None
        
    def __naive(self,Xf,XT,X0,X1,Y,I,I_C):
        self.model1.fit(np.column_stack((XT[I_C],Xf[I_C])),Y[I_C])
        g0 = self.model1.predict(np.column_stack((X0[I],Xf[I])))
        g1 = self.model1.predict(np.column_stack((X1[I],Xf[I])))
        return g0,g1
    
    def __ipw(self,Xf,T,I,I_C):
        self.model2.fit(Xf[I_C],T[I_C])
        m = self.model2.predict(Xf[I])
        return m
    
    def __fit_L(self,Xf,XT,T,X0,X1,Y,I,I_C):
        g0,g1 = self.__naive(Xf,XT,X0,X1,Y,I,I_C)
        m = self.__ipw(Xf,T,I,I_C)
        psi = (g1-g0) + T[I]*(Y[I]-g1)/m - (1-T[I])*(Y[I]-g0)/(1-m)
        beta_hat = np.mean(psi)
        
        return beta_hat
        
        
    def fit(self,X,T,Y,L=5,basis=False,standardize=False):
        self.L = L
        
        self.N = len(Y)
        self.I_split = np.array_split(np.array(range(self.N)),L)
        X0 = np.repeat(0,self.N).reshape(self.N,1)
        X1 = np.repeat(1,self.N).reshape(self.N,1)
        XT = T
        Xf = np.array((X))
        beta_hat = np.zeros(self.L)
        for i in range(L):
            if L==1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I=self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.N) if x not in I]
                
            beta_hat[i] = self.__fit_L(Xf,XT,T,X0,X1,Y,I,I_C)
        self.ATE = np.mean(beta_hat)
            
    def __augment(self,X,T,ind=None):
        T = T.reshape(len(T),1)
        XT= np.column_stack((T,(T**2),(T**3),T*X))
        Xf = np.column_stack((X,X**2,X**3))
        Xf = np.unique(Xf,axis=1)
        if np.array_equal(ind,None):
            XT,ind = np.unique(XT,axis=1,return_index=True)
        else: 
            XT = XT[:,ind]
        return XT, Xf, ind
    
    # This function is used to scale, but only non-dummy variables are
    # re-scaled. 
    def __scale_non_dummies(self,D,scaler=None):
        D = pd.DataFrame(D)
        if scaler==None:
            scaler = sklearn.preprocessing.StandardScaler()  
            D[D.select_dtypes('float64').columns] = scaler.fit_transform(D.select_dtypes('float64')) 
        else:
            D[D.select_dtypes('float64').columns] = (D[D.select_dtypes('float64').columns]-scaler.mean_)/scaler.scale_
        return np.array(D), scaler
    
    # This function makes sure all the data and inputs are in the right format 
    # before fitting. The data is scaled
    def __reformat(self,X,T,Y,standardize):
        if standardize==True:
            df = pd.DataFrame(data = np.column_stack((Y,T,X)))
            self.scaling = {'mean_Y':np.mean(df[0]),
                     'sd_Y':np.std(df[0]),
                     'mean_T':np.mean(df[1]),
                     'sd_T':np.std(df[1])}
            df[df.select_dtypes('float64').columns] = sklearn.preprocessing.StandardScaler().fit_transform(df.select_dtypes('float64'))
            
            Y = df[0]
            T = df[1]
            X = df.loc[:,2:]
            del df

        X = np.array((X))
        T = np.array((T))
        Y = np.array((Y))
        return X,T,Y,t_list
    
    # This function is used at the end of estimation to convert estimates into
    # numbers that are interpretable based on the scale of the original data-set
    def __descale(self):
        self.std_errors = self.std_errors*self.scaling['sd_Y']
        self.beta = (self.beta*self.scaling['sd_Y']) +self.scaling['mean_Y']
        
        
        
        
    