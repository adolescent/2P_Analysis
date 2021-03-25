# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:28:17 2021

@author: ZR
"""


def Stim_ID_Combiner(mode,para_dic = None):
    
    Stim_IDs = {}
    if mode == 'G16_Oriens':
        print('Orientation 0 is horizontal, counterclockwise.')
        Stim_IDs['Orien0'] = [1,9]
        Stim_IDs['Orien22.5'] = [2,10]
        Stim_IDs['Orien45'] = [3,11]
        Stim_IDs['Orien67.5'] = [4,12]
        Stim_IDs['Orien90'] = [5,13]
        Stim_IDs['Orien112.5'] = [6,14]
        Stim_IDs['Orien135'] = [7,15]
        Stim_IDs['Orien157.5'] = [8,16]
        Stim_IDs['Blank'] = [0]
        
    elif mode == 'G16_Dirs':
        print('Dir 0 is horizontal, moving up. Counterclockwise.')
        for i in range(16):
            current_name = 'Dir'+str(i*22.5)
            Stim_IDs[current_name] = [i+1]
        Stim_IDs['Blank'] = [0]
    return Stim_IDs


