# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:21:26 2021

@author: ZR
"""
from PIL import ImageGrab
import Graph_Operation_Kit as gt
import numpy as np
import os
import time

#%% Initializing
f = open('Config.txt','r')
config_info = f.readlines()
del f
L,U = config_info[3].split(',')
L = int(L)
U = int(U)
Height = int(config_info[6])
Width = int(config_info[9])
cap_freq = float(config_info[12])
sleep_time = 1/cap_freq
avr_frame = int(config_info[15])

img = np.array(ImageGrab.grab(bbox=(L, U, Width, Height)))[:,:,[2,1,0]]
gt.Show_Graph(img, 'Reference',os.getcwd(),0)
print('Screen Capture Averaging...')

#%% Capturing
c_frame = 0
cycle_num = 0
averaged_graph = np.zeros(shape = img.shape,dtype = 'f8')
global_average = np.zeros(shape = img.shape,dtype = 'f8')
while 1 :
    c_img = np.array(ImageGrab.grab(bbox=(L, U, Width, Height)))[:,:,[2,1,0]]
    if c_frame<avr_frame:
        averaged_graph += c_img/avr_frame
        time.sleep(sleep_time)
        c_frame +=1
    else:
        global_average = global_average*cycle_num/(cycle_num+1)+averaged_graph/(cycle_num+1)
        averaged_graph = np.clip(averaged_graph,0,255).astype('u1')
        gt.Show_Graph(averaged_graph,'Cycle_'+str(cycle_num),os.getcwd(),0)
        global_average = np.clip(global_average,0,255).astype('u1')
        gt.Show_Graph(global_average,'Global_Average',os.getcwd(),0)
        global_average = global_average.astype('f8')
        c_frame = 0
        cycle_num +=1
        averaged_graph = np.zeros(shape = img.shape,dtype = 'f8')
        print('Graph '+ str(cycle_num)+' generated')