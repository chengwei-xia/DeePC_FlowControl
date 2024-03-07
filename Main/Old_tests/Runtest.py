# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:28:55 2022

This script handles the whole process of implementing date-enabled predictive control
to design a controller for controlling flow around a rectangular cylinder at low Re

@author: xcw
"""

import os
import numpy as np
import time 
import matplotlib as mpl
import csv

import sys
# sys.path.append('/home/issay/UROP/DPC_Flow/Utils')
# sys.path.append('/home/issay/UROP/DPC_Flow/Environment')
# sys.path.append('/home/issay/UROP/DPC_Flow/mesh')

from GetHankels import Hankelbuilder,GetHankels
from DPC_Controller import DPC_Controller
from env import resume_env, nb_actuations, simulation_duration
from SaveResults import save_pred

# Copied from PyDeePC
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc import DeePC
from pydeepc.utils import Data
from utils import System

# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.ones(y.shape)
    return  cp.norm(y-ref,'fro')**2

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    # no real constraints on y, input should be between -1 and 1
    return [u >= -1, u <= 1]

# DeePC paramters
s = 1                       # How many steps before we solve again the DeePC problem
T_INI = 2                   # Size of the initial set of data
T_tr = 100
T_list = [T_tr]              # Number of data points used to estimate the system
HORIZON = 10                # Horizon length
LAMBDA_G_REGULARIZER = 0    # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0    # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0    # u regularizer
EXPERIMENT_HORIZON = 100    # Total number of steps

###### Setup parameters ###### 

lam_g = 0.5 # Regularization on g
lam_y = 0.5 # Regularization on sigmay
dt = 0.004 # Time step
t_c = 0.5 # Sampling time of control action, the number of numerical steps is t_c/dt = 125, consistent with RL
sim_tr = 200 # Non-dimensional training time. For numerical steps, use steps = sim_tr/dt
sim_run = 200 # Non-dimensional running time 
num_u = 1 # The number of control actions, e.g. only 1 mass flow rate is needed for jets
# num_y = 4 # The number of measurements, e.g. 64 sensors for pressure or 32 for antisymmetric pressure measurements
num_g = T_tr-T_INI-HORIZON+1 # Dimension of g or the width of final Hankel matrix

## Setup the class of flow environment
Flow_environment = resume_env(plot=False, single_run=True, dump_debug=1) 
action_step_size = simulation_duration / nb_actuations # Use action_step_size instead of t_c
num_y = len(Flow_environment.output_params["locations"])

## Initialize matrices to store data
num_action = int(T_tr/action_step_size)
utr = np.zeros([num_u,num_action])
ytr = np.zeros([num_y,num_action])
# urun = np.zeros([num_u,num_action])
# yrun = np.zeros([num_y,num_action])

##### Data Collection (Parallelization needs development) ######

## Reset flow to steady vortex shedding
Flow_environment.reset()
print("Reset environment for training")

## Random inputs for training, should be rich and long enough
num_action = int(sim_tr/action_step_size)
print("Sim_tr: " + str(sim_tr) + " action_step_size: "+ str(action_step_size) + " num_action: "+   str(num_action))
w = 0.01*np.random.rand(num_action,) # Input with noise
x = np.linspace(0,num_action, num_action+1) # Input with multiple frequency components
uran = 0.05* ( np.sin(100*np.pi*x/num_action) +  np.sin(10*np.pi*x/num_action) + np.sin(50*np.pi*x/num_action) )

# Simulate for different values of T
for T in T_list:
    print(f'Simulating with {T} initial samples...')
    
    
    ## Run the simulation with excitations
    Q = w[T]+uran[T]
    action = np.array([Q,-Q],)
    # Env(Qs) -- or build an environment to handle simulation and measurement from probes
    y,step = Flow_environment.execute(action) # 125 numerical steps, 0.5 nondimensional time
    
    ## Obtain training data
    utr[0,T] = Q
    
    for num_probes in range(num_y):
        ytr[num_probes,T]=(y[num_probes])

    print("One action step applied. On action step ", T+1, "out of ", num_action, ".", " Total simulation step = ", step)
        
    # Generate initial data and initialize DeePC
    data = np.vstack(utr,ytr)# Form a data matrix from simulation
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
    #sys.reset(data_ini = data_ini)

    deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = LAMBDA_G_REGULARIZER,
        lambda_y = LAMBDA_Y_REGULARIZER,
        lambda_u = LAMBDA_U_REGULARIZER)

    for _ in range(EXPERIMENT_HORIZON//s):
        # Solve DeePC
        u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True)

        # Apply optimal control input
        _ = sys.apply_input(u = u_optimal[:s, :], noise_std=1e-2)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    # data = sys.get_all_samples()
    # ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')
    # ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')

###### Training ###### 

test = False
## Load offline data, otherwise train again
filename = "Hankel.npz" ## File type to be decided
if(not os.path.exists("Saved_Hankels")):
    os.mkdir("Saved_Hankels")
# if(test):
#     os.remove("Saved_Hankels")
if(os.path.exists("Saved_Hankels/"+filename)):
    ##### Load
    Hankel = np.load("Saved_Hankels/"+filename)
    Up = Hankel['Up']
    Uf = Hankel['Uf']
    Yp = Hankel['Yp']
    Yf = Hankel['Yf']
    Uini = Hankel['Uini']
    Yini = Hankel['Yini']
    print('Hankel is loaded.')
    
elif(test):
    utr = np.random.rand(1,20)
    ytr = np.random.rand(4,20)
    print("Finish training.")

    
    ## Build Hankel matrices
    Up,Uf,Yp,Yf,Uini,Yini = GetHankels(utr,ytr,Tf,Tini,T)
    Uini = Uini.flatten()
    Yini = Yini.flatten()
    
    
    print("Hankel is generated.")
    
    ##### Write Hankel matrices to store offline data
    np.savez("Saved_Hankels/"+filename, Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, Uini=Uini, Yini=Yini)
    print("Hankel is saved (test).")
else:
    ##### Do training (Parallelization needs development)
    ## Reset flow to steady vortex shedding
    Flow_environment.reset()
    print("Reset environment for training")
    
    ## Random inputs for training, should be rich and long enough
    num_action = int(sim_tr/action_step_size)
    print("Sim_tr: " + str(sim_tr) + " action_step_size: "+ str(action_step_size) + " num_action: "+   str(num_action))
    w = 0.01*np.random.rand(num_action,) # Input with noise
    x = np.linspace(0,num_action, num_action+1) # Input with multiple frequency components
    uran = 0.05* ( np.sin(100*np.pi*x/num_action) +  np.sin(10*np.pi*x/num_action) + np.sin(50*np.pi*x/num_action) )
    


    for k in range(num_action): # for number of actions num_action
        ## Run the simulation with excitations
        Q = w[k]+uran[k]
        action = np.array([Q,-Q],)
        # Env(Qs) -- or build an environment to handle simulation and measurement from probes
        y,step = Flow_environment.execute(action) # 125 numerical steps, 0.5 nondimensional time
        
        ## Obtain training data
        utr[0,k] = Q
        ## NEED TEST!
        for num_probes in range(num_y):
            ytr[num_probes,k]=(y[num_probes])

        print("One action step applied. On action step ", k+1, "out of ", num_action, ".", " Total simulation step = ", step)
        
        
        
    print("Finish training.")
    
    ytr_write = open('ytr.csv', 'w+', newline = '')
    with ytr_write:
        write = csv.writer(ytr_write)
        write.writerows(ytr) 
    
    ## Build Hankel matrices
    Up,Uf,Yp,Yf,Uini,Yini = GetHankels(utr,ytr,Tf,Tini,T)
    
    print("Hankel is generated.")
    ##### Write Hankel matrices to store offline data
    np.savez("Saved_Hankels/"+filename, Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, Uini=Uini, Yini=Yini)
    print("Hankel is saved.")



###### Running ######

## Run DeePC control to obtain the first dumpy actions
Uo,Yo,g = DPC_Controller(Up,Uf,Yp,Yf,Uini,Yini,T,Tf,Tini,num_y,num_u,sparse=True)
save_pred(step,Yo,name = "Y_predict")
save_pred(step,Uo,name = "U_predict")
## Reset flow to steady vortex shedding
Flow_environment.reset()
print("Reset environment for running")


###### Postprocessing ######

