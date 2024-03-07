import numpy as np
from numpy.linalg import inv as mat_inv
from numpy.linalg import lstsq as left_divide
from scipy.linalg import hankel as hankel
from scipy.signal import StateSpace
from scipy.signal import cont2discrete
#from scipy.signal import tf2ss as tf2ss
import math
import cmath
import control as ct
import control.matlab as ctmat
from DPC_Controller import DPC_Controller


#Defining system
m = 0.5 #Mass
c = 0.5 #Damping Constant
k = 3 #Spring Constant
T_sim = 20 #Sim duration
T = 40
Tini = 5 #Initial data duration
Tf = 15 #Prediction horizon
force = 2 #Maximum focing input
T_tr = T_sim*2 #Training duration
dt = 1

A = np.array([[-c/m, -k/m],[1,0]])
B = np.array([[1],[0]])
C = np.array([[0,1]])
D = np.array([[0]])

sys_c = ctmat.ss(A,B,C,D)
sys = ctmat.c2d(sys_c,dt)

utr = force* ((np.random.rand(T_tr,)*2)-1)
usim = force* ((np.random.rand(T,)*2)-1)

Ttr_array = np.arange(0,T_tr,dt)
T_array = np.arange(0,T,dt)

#x0 = np.zeros((2,1))
x0 = np.array([[0,1]])
ytr, t_train, x_train = ctmat.lsim(sys,utr,Ttr_array)
ysim, t_sim, x_sim = ctmat.lsim(sys,usim,T_array)

H_u = hankel(utr[0:T_sim],utr[T_sim-1:T])
H_y = hankel(ytr[0:T_sim],ytr[T_sim-1:T])

Up = H_u[0:Tini,:]
Uf = H_u[Tini:,:]
Yp = H_y[0:Tini,:]
Yf = H_y[Tini:,:]

Uini = usim[0:Tini]
Yini = ysim[0:Tini]

Uo, Yo, g = DPC_Controller(Up,Uf,Yp,Yf,Uini,Yini,T,Tf,Tini,1,1, sparse= True)

