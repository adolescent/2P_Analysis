'''
This part will we intrapolate example loc's stim map, and compare it with cutted OI map.
'''

#%% Load in
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings("ignore")


from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot


expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
# c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')

#%%
'''
Step1 Get Response Frame, and define filter function.
'''
ac_locs = ac.Cell_Locs
od_resp = ac.OD_t_graphs['OD'].loc['CohenD']
hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD']
ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD']
red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD']

# Fill cell location with given color
def Graph_Filler(cell_locs,cell_resp,extend = 50,clip = 1):
    response_frame = np.zeros(shape = (512,512),dtype = 'f8')
    clipped_cell_resp = np.clip(cell_resp,-cell_resp.std()*clip,cell_resp.std()*clip)
    for i,c_response in enumerate(clipped_cell_resp):
        y_cord,x_cord = ac_locs[i+1]
        # y_min = max(y_cord-extend,0)
        # y_max = min(y_cord+extend,511)
        # x_min = max(x_cord-extend,0)
        # x_max = min(x_cord+extend,511)
        x = np.arange(512)
        y = np.arange(512)
        X, Y = np.meshgrid(x, y)
        gaussian = c_response*np.exp(-((X-x_cord)**2+(Y-y_cord)**2)/(2*extend**2))
        # response_frame[y_min:y_max,x_min:x_max] = c_response
        response_frame += gaussian

    return response_frame

hv_frame = Graph_Filler(ac_locs,hv_resp,50,1)
ao_frame = Graph_Filler(ac_locs,ao_resp,50,1)
od_frame = Graph_Filler(ac_locs,od_resp,50,1)
red_frame = Graph_Filler(ac_locs,red_resp,50,1)

hv_cell_frame = ac.Generate_Weighted_Cell(hv_resp)
ao_cell_frame = ac.Generate_Weighted_Cell(ao_resp)
od_cell_frame = ac.Generate_Weighted_Cell(od_resp)
red_cell_frame = ac.Generate_Weighted_Cell(red_resp)

#%% Plot filted 2P graph
from scipy.ndimage import gaussian_filter
c_graph = -red_frame
filted_graph = c_graph-gaussian_filter(c_graph, sigma=150)

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 300)
sns.heatmap(filted_graph,cmap='gist_gray',center = 0,xticklabels=False,yticklabels=False,square=True,cbar= None,ax = ax)

#%% Plot raw 2p graph
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 300)
sns.heatmap(-red_cell_frame,cmap='gist_gray',center = 0,xticklabels=False,yticklabels=False,square=True,cbar= None,ax = ax)
