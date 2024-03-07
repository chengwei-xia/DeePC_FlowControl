# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:16:42 2022

@author: xcw

This script is to build Hankel matrix from the training data of a user defined environment

____Inputs____
trlength: The total number of training data samples
u: Data of control input for training
y: Data of measurement for training
Tf: Prediction horizon
Tini: Initialization horizon
T: The length of training samples used for Hankel matrix

____Outputs____
Up= The past control input (in Hankel matrix form)
Uf= The future control input 
Yp= The past measurement
Yf= The future measurement
"""

import numpy as np
#import math
from scipy.linalg import hankel

def GetHankels(u,y,Tf,Tini,T):
    
    #Load training data and initial data'
    utr = u[:,:T]
    ytr = y[:,:T]
    uini = uini = u[:,T-Tini:T] #uini = u[:,T-Tini:T] u[:,0:Tini]
    yini = yini = y[:,T-Tini:T] #yini = y[:,T-Tini:T] y[:,0:Tini]

    """
    Up = np.zeros([Tini,num_g])
    Uf = np.zeros([Tf,num_g])
    Yp = np.zeros([Tini,num_g])
    Yf = np.zeros([Tf,num_g])
    """
    num_u = np.shape(utr)[0] # The number of u in one sample
    num_y = np.shape(ytr)[0]
    Up,Uf = Hankelbuilder(utr,T,Tf,Tini,num_u)
    Yp,Yf = Hankelbuilder(ytr,T,Tf,Tini,num_y)
    
    return Up,Uf,Yp,Yf,uini,yini

def Hankelbuilder(data,T,Tf,Tini,num):
    
    num_g = T-Tini-Tf+1 #The second dimension of Hankel matrix
    hp = np.zeros([Tini*num,num_g]) # n is the dimension of your state
    hf = np.zeros([Tf*num,num_g])
    
    for j in range(num):
        pcol = data[j,:Tini]
        prow = data[j,Tini-1:T-Tf]
        fcol = data[j,Tini:Tini+Tf]
        frow = data[j,Tini+Tf-1:T]
        pj = hankel(pcol,prow) # get the intermediate hakel for the jjth dimension of the state
        fj = hankel(fcol,frow)
        sizepj = np.shape(pj)[0]
        sizefj = np.shape(fj)[0]
        
        for k in range(sizepj):
            hp[k*num+j,:] = pj[k,:] # allocate the intermediate hakel to the final large hankel
        
        for k in range(sizefj):
            hf[k*num+j,:] = fj[k,:]
        
    return hp,hf