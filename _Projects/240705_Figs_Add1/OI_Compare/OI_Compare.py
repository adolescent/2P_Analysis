'''
This will make OI graph to compare with stim map.
Graph already cut, so we only need to intersect them.

'''

#%%
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2

# from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
import random

wp = r'D:\_GoogleDrive_Files\#Figs\#Second_Final_Version-240710\Add1_OI_Compare'
ac = ot.Load_Variable(r'D:\_All_Spon_Data_V1\L76_18M_220902\Cell_Class.pkl')
#%%
# intersect oi graphs
ao_oi = cv2.imread(ot.join(wp,'AO_OI.png'),0)
ao_oi_inter = cv2.resize(ao_oi,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
cv2.imwrite(ot.join(wp,'AO_OI_Inter.bmp'),ao_oi_inter)
hv_oi = cv2.imread(ot.join(wp,'HV_OI.png'),0)
hv_oi_inter = cv2.resize(hv_oi,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
cv2.imwrite(ot.join(wp,'HV_OI_Inter.bmp'),hv_oi_inter)
od_oi = cv2.imread(ot.join(wp,'OD_OI.png'),0)
od_oi_inter = cv2.resize(od_oi,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
cv2.imwrite(ot.join(wp,'OD_OI_Inter.bmp'),od_oi_inter)
rg_oi = cv2.imread(ot.join(wp,'RGLum_OI.png'),0)
rg_oi_inter = cv2.resize(rg_oi,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
cv2.imwrite(ot.join(wp,'RGLum_OI_Inter.bmp'),rg_oi_inter)


#%% Get 2P example graph
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

hv_frame = Graph_Filler(ac_locs,hv_resp,40,1)
ao_frame = Graph_Filler(ac_locs,ao_resp,40,1)
od_frame = Graph_Filler(ac_locs,od_resp,40,1)
red_frame = Graph_Filler(ac_locs,red_resp,40,1)

hv_cell_frame = ac.Generate_Weighted_Cell(hv_resp)
ao_cell_frame = ac.Generate_Weighted_Cell(ao_resp)
od_cell_frame = ac.Generate_Weighted_Cell(od_resp)
red_cell_frame = ac.Generate_Weighted_Cell(red_resp)

# and we make hv frame into 20 bin.
hv_frame_inter = cv2.resize(hv_frame[20:492,20:492],(24,25),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
hv_frame_inter = cv2.resize(hv_frame_inter,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
ao_frame_inter = cv2.resize(ao_frame[20:492,20:492],(24,25),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
ao_frame_inter = cv2.resize(ao_frame_inter,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
od_frame_inter = cv2.resize(od_frame[20:492,20:492],(24,25),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
od_frame_inter = cv2.resize(od_frame_inter,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
red_frame_inter = cv2.resize(red_frame[20:492,20:492],(24,25),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
red_frame_inter = cv2.resize(red_frame_inter,(512,512),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
#%% Plot 2P and save 2p.
from scipy.ndimage import gaussian_filter
# c_graph = -od_frame_inter
c_graph = -red_cell_frame
filted_graph = c_graph-gaussian_filter(c_graph, sigma=150)

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 300)
sns.heatmap(filted_graph,cmap='gist_gray',center = 0,xticklabels=False,yticklabels=False,square=True,cbar= None,ax = ax)


