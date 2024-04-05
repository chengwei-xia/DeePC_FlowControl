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
#import matplotlib as mpl
import csv

import sys
cwd = os.getcwd()
sys.path.append("..")
sys.path.append(cwd + "/../Environment")
# sys.path.append('/home/issay/UROP/DPC_Flow/Utils')
# sys.path.append('/home/issay/UROP/DPC_Flow/Environment')
# sys.path.append('/home/issay/UROP/DPC_Flow/mesh')

from Utils.GetHankels import Hankelbuilder,GetHankels
#from DPC_Controller import DPC_Controller
#from env import resume_env, nb_actuations, simulation_duration
from Utils.SaveResults import save_pred

# Copied from PyDeePC
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint

from Utils.utils import Data
from DPC_cvxpy import DeePC
from Build_sys import System, GymSystem, FlowSystem

from Environment.env import resume_env, nb_actuations

# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = 0*np.ones(y.shape)
    return  5*cp.norm(y-ref,'fro')**2

def loss_gym(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.ones(y.shape)
    return  cp.norm(y-ref,'fro')**2

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    # no real constraints on y, input should be between -1 and 1
    return [u >= -0.1, u <= 0.1]

###### Setup parameters ###### 

s = 1                       # How many steps to apply predicted control in receding horizon, usually apply only one step
T_INI = 50                  # Size of the initial set of data
T_tr = 400
T_list = [T_tr]              # Number of data points used to estimate the system
HORIZON = 50               # Horizon length
LAMBDA_G_REGULARIZER = 10   # Regularization on g (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 10  # Regularization on sigmay (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 10   # Regularization on sigmau
LAMBDA_PROJ_REGULARIZER = 0
EXPERIMENT_HORIZON = 80    # Total number of steps
dim_u = 1 # The number of control actions, e.g. only 1 mass flow rate is needed for jets
dim_y = 1 # The number of measurements, e.g. 64 sensors for pressure or 32 for antisymmetric pressure measurements
num_g = T_tr-T_INI-HORIZON+1 # Dimension of g or the width of final Hankel matrix

folder_data = "Offline_Data_" + str(T_tr)
cano_qp = True
Use_offline_data = True

sys = FlowSystem()

fig, ax = plt.subplots(1,2)
plt.margins(x=0, y=0)

##### Data Collection (Parallelization needs development) ######

# Simulate for different values of T

for T in T_list:
    
    print(f'Simulating with {T} initial samples...')

    sys.reset()

    excitation_u = np.random.uniform(low=-0.1, high=0.1, size=(T,1))
    #np.random.normal(size=T).reshape((T, 1)) #np.sin(np.linspace(1,T,T).reshape((T, 1)) * np.pi / 180. )
    ## Initialize DeePC object with excitation data np.random.normal(size=T).reshape((T, 1))
    #np.random.seed(10)

    if Use_offline_data == True:
        filename = "Offline_data.npz" ## File type to be decided
        if(not os.path.exists(folder_data)):
            os.mkdir(folder_data)
# if(test):
#     os.remove("Saved_Hankels")
        if(os.path.exists(folder_data+"/"+filename)):
    ##### Load
            Offline_data = np.load(folder_data+"/"+filename)
            Offline_u = Offline_data['u']
            Offline_y = Offline_data['y']
            if  Offline_u.shape[0] != T_tr:
                os.remove(folder_data+"/"+filename)
            assert Offline_u.shape[0] == T_tr, "Wrong offline data length. Rerun the code to generate new data."
            print('Offline data are loaded.')
            data = Data(u = Offline_u, y = Offline_y)
            
        else:
            print('Start data collection for off-line Hankel and save data.')
            data = sys.apply_input(u = excitation_u, noise_std=0) #, noise_std=0) #np.sin(np.linspace(1,T,T).reshape((T, 1)) * np.pi / 180. )
            np.savez(folder_data+"/"+filename, u=data.u, y=data.y)
            print("Offline data is saved.")
    else:
        print('Start data collection for off-line Hankel.')
        data = sys.apply_input(u = excitation_u, noise_std=0) #, noise_std=0) #np.sin(np.linspace(1,T,T).reshape((T, 1)) * np.pi / 180. )
        
    #data = sys.apply_input(u = np.random.uniform(-3,3,T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)
    print('Finish data collection and off-line Hankel is ready.')
    
    ## Create initial data
    data_ini = Data(u = np.zeros((T_INI, dim_u)), y = np.zeros((T_INI, dim_y)))
    
    sys.reset(data_ini=data_ini)
   # Uini, Yini = np.transpose(data_ini.u[:T_INI]), np.transpose(data_ini.y[:T_INI])

    ## Build DeePC problem
    
    if cano_qp:
        deepc.qp_setup(
            lambda_g = LAMBDA_G_REGULARIZER,
            lambda_y = LAMBDA_Y_REGULARIZER,
            lambda_u = LAMBDA_U_REGULARIZER,
            u_low    = -0.1,
            u_up     = 0.1,
            yref     = 0.0)
    elif not cano_qp:
        deepc.opt_setup(
            build_loss = loss_callback,
            build_constraints = constraints_callback,
            lambda_g = LAMBDA_G_REGULARIZER,
            lambda_y = LAMBDA_Y_REGULARIZER,
            lambda_u = LAMBDA_U_REGULARIZER,
            lambda_proj = LAMBDA_PROJ_REGULARIZER)
        
    for k in range(EXPERIMENT_HORIZON//s):
        
        start_time = time.perf_counter()
        ## Run DeePC control to obtain the first set of actions
        if cano_qp:
            Uo, Yo = deepc.qp_solve(data_ini = data_ini, warm_start=True)
        elif not cano_qp:
            Uo, info = deepc.solve(data_ini = data_ini, warm_start=True)
        end_time = time.perf_counter()
        print('One optimization step takes :% s' % ((end_time - start_time)))
        
        # Apply optimal control input for one step
        _ = sys.apply_input(u = Uo[:s, :], noise_std=0)
        
        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

        print("One action step applied. On action step ", k+1, "out of ", EXPERIMENT_HORIZON//s, ".")
        
    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, T_f={HORIZON}$')
    ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, T_f={HORIZON}$')
        
    #sys.close()
    print("Finish DeePC.")

ax[0].set_ylim(-0.5, 0.5)
ax[1].set_ylim(-0.1, 0.1)
ax[0].set_xlabel('t')
ax[0].set_ylabel('y')
ax[0].grid()
ax[1].set_ylabel('u')
ax[1].set_xlabel('t')
ax[1].grid()
ax[0].set_title('Closed loop - output signal $y_t$')
ax[1].set_title('Closed loop - control signal $u_t$')
plt.legend(fancybox=True, shadow=True)
plt.show()

