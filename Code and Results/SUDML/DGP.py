# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:45:15 2020

@author: Kyle
"""
import pandas as pd
import numpy as np
from scipy.sparse import diags
from scipy.stats import norm
N=1000

def DGP1(N,delta):
    k=100
    rho=0.5
    d = np.array([rho*np.ones(k-1),np.ones(k),rho*np.ones(k-1)])
    offset = [-1,0,1]
    x_cov = diags(d,offset).toarray()
    x_mean = np.ones(k)
    X = np.random.multivariate_normal(x_mean,x_cov,N)
    epsilon = np.random.multivariate_normal([0,0],[[1,0],[0,1]],size=N)
    theta = np.array([1 for l in list(range(1,(k+1)))])
    theta = theta.reshape(k,1)
    p = norm.cdf((X@theta-(theta.T@x_mean))/(theta.T@x_cov@theta))
    T = np.random.binomial(1,p)
    
    theta2 = theta+delta
    
    beta = [1 + (1*delta)/2, 1 + (2*delta)/2]
    
    Y1 = beta[0]*T + X@theta + epsilon[:,0,None]
    Y2 = beta[1]*T + X@theta2 + epsilon[:,1,None]
    columns = ['Y1','Y2','T'] + ['x'+str(a) for a in range(0,k)]
    data = pd.DataFrame(np.column_stack((Y1,Y2,T,X)),columns = columns)
    
    return data
    