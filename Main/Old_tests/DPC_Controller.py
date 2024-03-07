

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:27:31 2022

_______Inputs________
Hankel matrices: Up Uf Yp Yf
Initial data: Uini Yini
Parameters: T Tf Tini num_y num_u sparse(whether the qp matrices are sparse)
@author: xcw
"""

import os
import numpy as np
#import quadprog
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
import csv

def DPC_Controller(Up,Uf,Yp,Yf,Uini,Yini,T,Tf,Tini,num_y,num_u, sparse:bool = False):
    
    ###### Set up control parameters ######

    Umax =  1 # Maximum actuation
    Ytol =  2 # Maximum tolerence of y
    lam_g = 0
    lam_y = 0 # Penalty on sigmay
    lam_u = 0 # Penalty on sigmau
    
    num_g = T-Tini-Tf+1 # Dimension of g or the width of final Hankel matrix
    Duini = num_u*Tini  # Dimension of Uini and Yini
    Dyini = num_y*Tini
    Duf = num_u*Tf  # Dimension of Uf and Yf
    Dyf = num_y*Tf 
    Q = 1*np.ones(Dyf)
    R = 1*np.ones(Duf)
    Hankel = np.vstack ([Up,Yp,Uf,Yf])
    inv_eq = np.dot(np.linalg.pinv(Hankel),Hankel)
    #M_pre = np.eye(num_g)-np.dot(np.linalg.pinv(Hankel),Hankel)
    M_pre = np.eye(num_g)-np.dot(np.transpose(Hankel),Hankel)
    M = np.dot(np.transpose(M_pre),M_pre)

    
    ###### Solve quadratic programming (minimize 1/2*xT*P*x+qT*x, subject to Gx<=h, Ax=b, lb<=x<=ub) ######
    ## x = [y; u; g; sigmay; sigmau]
    
    ## Formulate weight matrices P q
    P= np.zeros([Dyf+Duf+num_g+Dyini+Duini, Dyf+Duf+num_g+Dyini+Duini])
    P[0:Dyf,0:Dyf] = Q
    P[Dyf:Dyf+Duf,Dyf:Dyf+Duf] = R
    #P[Dyf+Duf:Dyf+Duf+num_g,Dyf+Duf:Dyf+Duf+num_g] = np.eye(num_g)
    P[Dyf+Duf:Dyf+Duf+num_g,Dyf+Duf:Dyf+Duf+num_g] = lam_g*M
    P[Dyf+Duf+num_g:Dyf+Duf+num_g+Dyini,Dyf+Duf+num_g:Dyf+Duf+num_g+Dyini] = lam_y*np.eye(Dyini)
    P[Dyf+Duf+num_g+Dyini:Dyf+Duf+num_g+Dyini+Duini,Dyf+Duf+num_g+Dyini:Dyf+Duf+num_g+Dyini+Duini] = lam_u*np.eye(Duini)

    '''with open("P.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(P)'''
    
    #q = np.hstack([np.zeros([Dyf+Duf,]),lam_g*np.ones(num_g),lam_y*np.zeros(Dyini),lam_u*np.zeros(Duini)])
    q = np.zeros(Dyf+Duf+num_g+Dyini+Duini)
    
    ## Formulate constraints (to do: make a function that generates the constraint matrices G,h,A,b)
    
    # Uf*g<=Umax; -Uf*g<=Umax
    G1_1 = np.hstack([np.zeros([Duf,Dyf+Duf]), Uf, np.zeros([Duf,Dyini+Duini])])
    G1_2 = np.hstack([np.zeros([Duf,Dyf+Duf]), -Uf, np.zeros([Duf,Dyini+Duini])])
    G2 = np.vstack([G1_1,G1_2])
    h2 = np.hstack([Umax*np.ones(Duf), Umax*np.ones(Duf)])

    # Yf*g<=Ytol_max; -Yf*g<=Ytol_max   
    G2_1 = np.hstack([np.zeros([Dyf,Dyf+Duf]), Yf, np.zeros([Dyf,Dyini+Duini])])
    G2_2 = np.hstack([np.zeros([Dyf,Dyf+Duf]), -Yf, np.zeros([Dyf,Dyini+Duini])])
    G1 = np.vstack([G2_1,G2_2])
    h1 = np.hstack([Ytol*np.ones(Dyf), Ytol*np.ones(Dyf)])
    
    # Up*g-sigmau = uini:
    A2 = np.hstack([np.zeros([Duini,Duf+Dyf]), Up, np.zeros([Duini,Dyini]), -1*np.eye(Duini)])
    b2 = Uini.flatten('F')
        
    # Yp*g-sigmay=yini;
    A1 = np.hstack([np.zeros([Dyini,Duf+Dyf]), Yp, -1*np.eye(Dyini), np.zeros([Dyini,Duini])])
    b1 = Yini.flatten('F')

    #
    A3 = np.hstack([np.ones([Dyini,Duf+Dyf]), np.zeros(np.shape(Yp)), np.zeros([Duini,Dyini]), np.zeros([Dyini,Duini])])
    b3 = np.hstack([Yf,Uf])
    b3 = b3.flatten('F')

    # -Umax <= U <= Umax
    # lb = np.vstack([np.zeros([Dyf,1]),-Umax,np.zeros([num_g+Dyini,1])])
    # ub = np.vstack([np.zeros([Dyf,1]),Umax,np.zeros([num_g+Dyini,1])])

    # Assign all the weight matrice and constraints to final matrices
    
    G = np.vstack([G1,G2])
    h = np.hstack([h1,h2])
    A = np.vstack([A1,A2,A3])
    b = np.hstack([b1,b2,b3])

    with open("A.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(A)

    
    if sparse:
        P = csc_matrix(P)
        G = csc_matrix(G)
        A = csc_matrix(A)
        solver = "osqp" # Solver installed: osqp, scs, quadprog. Solver candidates: cvxopt, ecos(sparse), gurobi(sparse), mosek(sparse), osqp(sparse), qpoases, qpswift(sparse), quadprog, scs(sparse)
    else:
        solver = "quadprog"
     
    optx = solve_qp(P, q, G, h, A, b, solver=solver)
    #optx = solve_qp(P, q, None, None, A, b, solver="osqp")

    print(optx)
   
    
    g = optx[Duf+Dyf:Duf+Dyf+num_g]

    '''with open("optx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(optx)

    with open("g.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(g)'''
    
    # Predicted solution
    Uo = np.dot(Uf,g)
    Yo = np.dot(Yf,g)
    
    return Uo,Yo,g