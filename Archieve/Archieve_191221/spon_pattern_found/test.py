# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:18:55 2019

@author: ZR
"""

import matplotlib.pyplot as plt
import numpy as np
import functions_cluster as pp

plt.plot(correlation_plots[14,:])

#%%
import pickle
def save_variable(variable,name):
        fw = open(name,'wb')
        pickle.dump(variable,fw)#保存细胞连通性质的变量。 
        fw.close()
        
        
def read_variable(name):
    with open(name, 'rb') as file:
        variable = pickle.load(file)
    file.close()
    return variable

cell_group = read_variable(r'G:\ZR\data_processing\190405_L74_LM\1-019\results\cell_group')
#%%
a =[]
for i in range(0,23):
    a.extend(Frame_Stim_Check['0'][i])