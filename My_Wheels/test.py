# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%% Use a as base variable.
b = (a>2).astype('i4')
import more_itertools as mit
c =list(mit.split_when(b,lambda x, y: (x-y) == -1))

#%%
processed_list = []
for i in range(len(c)):
    current_list = np.dot(c[i],i+1)-1
    processed_list.extend(current_list)    