# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:26:33 2022

@author: xcw
"""
import numpy as np
import os
import csv

def save_pred(step,data,name):
    ## Store and output control data
    filename = name+".csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+filename)):
        with open("saved_models/"+filename, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["CurrentStep", name])
            for i in range(len(data)):
                spam_writer.writerow([step,data[i]])
    else:
        with open("saved_models/"+filename, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            for i in range(len(data)):
                spam_writer.writerow([step,data[i]])
                
def save_run(step,data,name):
    ## Store and output control data
    filename = name+".csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+filename)):
        with open("saved_models/"+filename, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Step", name])
            spam_writer.writerow([step, data])
    else:
        with open("saved_models/"+filename, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow([step, data])    

