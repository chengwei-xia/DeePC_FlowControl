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


LAMBDA_G_REGULARIZER = np.array([1,10])   # Regularization on g (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = np.array([10])  # Regularization on sigmay (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = np.array([1])   # Regularization on sigmau
LAMBDA_PROJ_REGULARIZER = [1e-4,1e-3,1e-2]
s = 1                       # How many steps to apply predicted control in receding horizon, usually apply only one step
EXPERIMENT_HORIZON = 10   # Total number of steps
R = 0                      # Weight for R matrix in optimization (uT*R*u)
Q = 1e5                    # Weight for Q matrix in optimization (yT*Q*y)
umin = -0.1
umax = 0.1
yref = -0.62
solver = cp.SCS

T_INI = 150               # Size of the initial set of data
T_tr = 1000
T_list = [T_tr]              # Number of data points used to estimate the system
HORIZON = 150             # Horizon length
dim_u = 1 # The number of control actions, e.g. only 1 mass flow rate is needed for jets
dim_y = 1 # The number of measurements, e.g. 64 sensors for pressure or 32 for antisymmetric pressure measurements
Hankel_up_steps = 50 # The number of steps after which we update online Hankel matrices

folder_data = "Offline_Data_" + str(T_tr) + "_drag"

cano_qp = True
Use_offline_data = True
Online_hankel = False


def DeePC_Run(data_path, opt_params, hankel_params, selectors):
    
    ##### Path Variables #####
    folder_data = data_path['offline_data']
    
    ##### Optimization Variables #####
    LAMBDA_G_REGULARIZER = opt_params['lambda_g']  
    LAMBDA_Y_REGULARIZER = opt_params['lambda_y']  # Regularization on sigmay (see DeePC paper, eq. 8)
    LAMBDA_U_REGULARIZER = opt_params['lambda_u']   # Regularization on sigmau
    LAMBDA_PROJ_REGULARIZER = opt_params['lambda_proj']
    s = opt_params['control_applied']                       # How many steps to apply predicted control in receding horizon, usually apply only one step
    EXPERIMENT_HORIZON = opt_params['control_horizon'] 
    Weight_u = opt_params['R']
    Weight_y = opt_params['Q']
    umin     = opt_params['umin']
    umax     = opt_params['umax']
    yref     = opt_params['yref']
    solver   = opt_params['solver']
    
    
    ##### Hankel Matrix Variables #####
    T = hankel_params['T']
    T_INI = hankel_params['Tini']              
    HORIZON = hankel_params['Tf']             
    dim_u = hankel_params['dim_u']  
    dim_y = hankel_params['dim_y'] 
    num_g = T_tr-T_INI-HORIZON+1 # Dimension of g or the width of final Hankel matrix
    Hankel_ctr = 0
    
    ##### Selectors #####
    cano_qp = selectors['QP']
    Use_offline_data = selectors['Use_offline_data']
    Online_hankel = selectors['Online_hankel']
    
    
    mse = 0
    total_time = 0
    
    sys = FlowSystem()
##### Data Collection (Parallelization needs development) ######

# Simulate for different values of T
    
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
            u_low    = umin,
            u_up     = umax,
            yref     = yref,
            W_R      = Weight_u,
            W_Q      = Weight_y)
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
            Uo, Yo = deepc.qp_solve(data_ini = data_ini, warm_start=True, solver = solver)
        elif not cano_qp:
            Uo, info = deepc.solve(data_ini = data_ini, warm_start=True , solver = solver)
        end_time = time.perf_counter()
        print('One optimization step takes :% s' % ((end_time - start_time)))
        
        total_time += end_time - start_time
        # Apply optimal control input for one step
        _ = sys.apply_input(u = Uo[:s, :], noise_std=0)
        
        # Update hankel matrices
        if Online_hankel == True and k*s>=T:
            
            if Hankel_ctr%Hankel_up_steps==0:
                new_data = sys.get_last_n_samples(T)
                deepc.update_Hankel(data=new_data)
                Hankel_times = Hankel_ctr/Hankel_up_steps+1
                print(f'Update online Hankel for the {Hankel_times} time.')
            Hankel_ctr +=1
        #sum_reward += reward
        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)
        
        print("One action step applied. On action step ", k+1, "out of ", EXPERIMENT_HORIZON//s, ".")
        
    data = sys.get_all_samples()
    
    mse = np.mean((data.y[-20:]-yref)**2) #mse for last 20 steps
    
    info = {'T':T_tr,
            'Tini':T_INI,
            'Tf':HORIZON,
            'lambda_g': LAMBDA_G_REGULARIZER,
            'lambda_y': LAMBDA_Y_REGULARIZER,
            'lambda_u': LAMBDA_U_REGULARIZER,
            'R': R,
            'Q': Q,
            'solver':solver,
            'avg_opt_step_time':total_time/(EXPERIMENT_HORIZON//s),
            'total_steps':EXPERIMENT_HORIZON,
            'mse':mse
            }

        
    return data, info #, sum_reward

def log_params(info):
    
    print("Start parameter logging.")
    name = "log_params.csv"
    if (not os.path.exists("logging")):
        os.mkdir("logging")
    if (not os.path.exists("logging/" + name)):
        with open("logging/" + name, "w") as csv_obj:
            spam_writer = csv.writer(csv_obj, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["T" ,"Tini" ,"Tf" ,"lambda_u" ,"lambda_y" ,"lambda_g" ,"Q" ,"R" ,"solver" ,"avg_opt_step_time", "total_steps", "mse"])
            spam_writer.writerow([info['T'], info['Tini'], info['Tf'],info['lambda_u'],info['lambda_y'],info['lambda_g'],info['Q'],info['R'],info['solver'],info['avg_opt_step_time'],info['total_steps'],info['mse']])
    else:
        with open("logging/" + name, "a") as csv_state:
            spam_writer = csv.writer(csv_state, delimiter=";", lineterminator="\n")
            spam_writer.writerow([info['T'], info['Tini'], info['Tf'],info['lambda_u'],info['lambda_y'],info['lambda_g'],info['Q'],info['R'],info['solver'],info['avg_opt_step_time'],info['total_steps'],info['mse']])
    print("Finish parameter logging.")
    

#### Main code for running ####
data_path = {'offline_data': folder_data}


    
hankel_params = {'T':T_tr,
                 'Tini':T_INI,
                 'Tf':HORIZON,
                 'dim_u':dim_u,
                 'dim_y':dim_y
                 }

selectors = {'QP':cano_qp,
             'Use_offline_data':Use_offline_data,
             'Online_hankel':Online_hankel
             }


for i in range(len(LAMBDA_G_REGULARIZER)):
    for j in range(len(LAMBDA_Y_REGULARIZER)):
        for k in range(len(LAMBDA_U_REGULARIZER)):
            
            # Set up parameters
            opt_params = {'lambda_g': LAMBDA_G_REGULARIZER[i],
                          'lambda_y': LAMBDA_Y_REGULARIZER[j],
                          'lambda_u': LAMBDA_U_REGULARIZER[k],
                          'lambda_proj': LAMBDA_PROJ_REGULARIZER,
                          'control_applied': s,
                          'control_horizon': EXPERIMENT_HORIZON,
                          'R': R,
                          'Q': Q,
                          'umin':umin,
                          'umax':umax,
                          'yref':yref,
                          'solver':solver
                          }
            
            # Run control experiments
            data,info = DeePC_Run(data_path=data_path, opt_params=opt_params, hankel_params=hankel_params, selectors=selectors)
            
            # Save parameter logs
            log_params(info)
            
            # Plot curve
            fig, ax = plt.subplots(1,2)
            plt.margins(x=0, y=0)
            
            ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T_tr}, T_i={T_INI}, T_f={HORIZON}$')
            ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T_tr}, T_i={T_INI}, T_f={HORIZON}$')

            ax[0].set_ylim(-0.6, -0.8)
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
            fig_name = f'drag_T_{T_tr}_Tini_{T_INI}_Tf_{HORIZON}_g_' + "{:.1f}".format(LAMBDA_G_REGULARIZER[i]).replace('.','p') + '_y_' + "{:.1f}".format(LAMBDA_Y_REGULARIZER[j]).replace('.','p') + '_u_' + "{:.1f}".format(LAMBDA_U_REGULARIZER[k]).replace('.','p')
            plt.savefig(fname = fig_name)
            plt.show()
            
            y_mse = np.mean(data.y[-10:,:]**2)

    #sys.close()
print("Finish DeePC.")
#if test_hankel_params == True:
    #for i in range(len(LAMBDA_G_REGULARIZER)):
