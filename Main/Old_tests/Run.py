# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:28:55 2022

This script handles the whole process of implementing date-enabled predictive control
to design a controller for controlling flow around a rectangular cylinder at low Re

@author: xcw
"""
import sys
sys.path.append('/home/issay/UROP/DPC_Flow/Utils')
sys.path.append('/home/issay/UROP/DPC_Flow/Environment')
sys.path.append('/home/issay/UROP/DPC_Flow/mesh')

import os
import numpy as np
import time 
#import matplotlib.pyplot as plt
import csv

from GetHankels import Hankelbuilder,GetHankels
from DPC_Controller import DPC_Controller
from env import resume_env, nb_actuations, simulation_duration
from SaveResults import save_pred, save_run

###### Setup parameters ###### 

T = 40 # Length of training data collection
Tf = 20 # Horizon of prediction
Tf_run = 3 # Horizon of applying control actions. Should be smaller than Tf that only Tf_run actions are picked in the Tf actions
Tini = 10 # Length of initial data
dt = 0.004 # Time step
t_c = 0.5 # Sampling time of control action, the number of numerical steps is t_c/dt = 125, consistent with RL
sim_tr = 30 # Non-dimensional training time. For numerical steps, use steps = sim_tr/dt
sim_run = 10 # Non-dimensional running time 
num_u = 1 # The number of control actions, e.g. only 1 mass flow rate is needed for jets
num_y = 1 # The number of measurements, e.g. 64 sensors for pressure or 32 for antisymmetric pressure measurements
num_g = T-Tini-Tf+1 # Dimension of g or the width of final Hankel matrix

## Setup the class of flow environment
Flow_environment = resume_env(plot=False, single_run=True, dump_debug=1) 
action_step_size = simulation_duration / nb_actuations # Use action_step_size instead of t_c. Usually action_step_size = 0.5 (non-dimensional time)
num_Y = len(Flow_environment.output_params["locations"])

## Initialize matrices to store data
utr = np.zeros([num_u,int(sim_tr/action_step_size)])
ytr = np.zeros([num_y,int(sim_tr/action_step_size)])
# urun = np.zeros([num_u,int(sim_run/action_step_size)])
# yrun = np.zeros([num_y,int(sim_run/action_step_size)])
urun = np.zeros([num_u,Tf])
yrun = np.zeros([num_y,Tf])

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
    
    # if num_action < T:
    #     assert("Not enough training actions.")
    
    w = 0.3*np.random.rand(num_action+1,) # Input with noise
    x = np.linspace(0,num_action, num_action+1) # Input with multiple frequency components
    #uran = 0.3* ( np.sin(100*np.pi*x/num_action) +  np.sin(10*np.pi*x/num_action) + np.sin(50*np.pi*x/num_action) )
    #uran = 0.02* ( np.sin(2*np.pi*x/13.7) +  np.sin(1*np.pi*x/13.7) + np.sin(4*np.pi*x/13.7) + np.sin(6*np.pi*x/13.7) )
    uran = 0.1* ((np.random.rand(num_action+1,)*2)-1)

    ## plot the excitation signal
    # plt.plot(x,(uran+w))
    # plt.xlabel("Number of actions")
    # plt.ylabel("Amplitude")
    # plt.title("Excitation inputs")

    '''output_y = []
    input_u = []'''

    for k in range(num_action): # for number of actions
        
        ## Run the simulation with excitations
        Q = uran[k]
        action = np.array([Q,-Q],)
        # Env(Qs) -- or build an environment to handle simulation and measurement from probes
        y,step = Flow_environment.execute(action) # 125 numerical steps, 0.5 nondimensional time
        
        '''output_y.extend(output_for_me)
        input_u.extend(input_for_me)'''
        
        print("One action step applied. On action step ", k+1, "out of ", num_action, ".", " Total simulation step = ", step)

        
        ## Obtain training data
        utr[0,k] = Q
        ## NEED TEST!
        for num_probes in range(num_y):
            ytr[:,k]=(y)
        
    print("Finish training.")
    with open("ytr.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(ytr)

    with open("utr.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(utr)

    '''with open("output.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(output_y)

    with open("input.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(input_u)'''


    ## Build Hankel matrices
    Up,Uf,Yp,Yf,Uini,Yini = GetHankels(utr,ytr,Tf,Tini,T) # Here Uini and Yini are the initial data just after steady vortex shedding. See DPC_Controller.py for details.
    print("Hankel is generated.")
    
    ##### Write Hankel matrices to store offline data
    np.savez("Saved_Hankels/"+filename, Up=Up, Uf=Uf, Yp=Yp, Yf=Yf, Uini=Uini, Yini=Yini)
    print("Hankel is saved.")


###### Running ######

## Run DeePC control to obtain the first dumpy actions
Uo,Yo,g = DPC_Controller(Up,Uf,Yp,Yf,Uini,Yini,T,Tf,Tini,num_y,num_u,sparse=True)
#save_pred(0,Yo,name = "Y_predict")
#save_pred(0,Uo,name = "U_predict")
## Reset flow to steady vortex shedding
Flow_environment.reset()
print("Reset environment for running")

num_action = int(sim_run/action_step_size/Tf_run)
for k in range(int(sim_run/action_step_size/Tf_run)): # for the number of horizons
    
#### Apply control actions and run the following control process

    ### For one control horizon, run Tf actions
    for i in range(Tf_run): 
    
        ## Run the simulation with optimal control actions
        Qs = Uo[i]
        action = np.array([Qs,-Qs],)
        y,step = Flow_environment.execute(action)
        print("One action step applied for running. Total simulation step = ", step)
        ## Save running data
        urun[0,i] = Qs
        for num_probes in range(num_y):
            yrun[:,i]=(y)
            
        save_run(step,y,"Y_run")
        save_run(step,Qs,"U_run")

        print("Running data collected")
        print("On control action ",i, " out of ", Tf_run)
    ###    
    ## Update initial data with the last Tini values
    Usave = np.hstack([Uini,urun])
    Ysave = np.hstack([Yini,yrun])
    Uini = Usave[:,-Tini:]
    Yini = Ysave[:,-Tini:]
    print(Yini)

    # Urun_save = urun[:,k*Tf:(k+1)*Tf].flatten()
    # Yrun_save = yrun[:,k*Tf:(k+1)*Tf].flatten()
    #Step = Flow_environment # Obtain the current numerical step
    
    print("Finish running for one horizon")
    
## Obtain new action after every horizon
    start = time.process_time()
    Uo,Yo,g = DPC_Controller(Up,Uf,Yp,Yf,Uini,Yini,T,Tf,Tini,num_y,num_u,sparse=True)
    end = time.process_time() 
    print("The function run time is : %s seconds" %(end-start))
    save_pred(step,Yo,name = "Y_predict")
    save_pred(step,Uo,name = "U_predict")
    print("Action prediction is updated for the next horizon.")

    print("One action step applied. On action step ", k+1, "out of ", num_action, ".", " Total simulation step = ", step)

print("Control finished")

#start = time.process_time()
#end = time.process_time()
#print("The function run time is : %.03f seconds" %(end-start))


###### Postprocessing ######