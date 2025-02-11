'''
This script will shuffle example location's orientation maps for 100 times, give us example of 100 set random graphs.
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
'''
Part 1, generate blured Orientation graph
'''
ac_locs = ac.Cell_Locs

o0_resp = ac.Orien_t_graphs['Orien0-0'].loc['CohenD']
o22_resp = ac.Orien_t_graphs['Orien22.5-0'].loc['CohenD']
o45_resp = ac.Orien_t_graphs['Orien45-0'].loc['CohenD']
o67_resp = ac.Orien_t_graphs['Orien67.5-0'].loc['CohenD']
o90_resp = ac.Orien_t_graphs['Orien90-0'].loc['CohenD']
o112_resp = ac.Orien_t_graphs['Orien112.5-0'].loc['CohenD']
o135_resp = ac.Orien_t_graphs['Orien135-0'].loc['CohenD']
o157_resp = ac.Orien_t_graphs['Orien157.5-0'].loc['CohenD']

def Graph_Filler(cell_resp,extend = 40,clip = 5):
    response_frame = np.zeros(shape = (512,512),dtype = 'f8')
    clipped_cell_resp = np.clip(cell_resp,cell_resp.mean()-cell_resp.std()*clip,cell_resp.mean()+cell_resp.std()*clip)
    for i,c_response in enumerate(clipped_cell_resp):
        y_cord,x_cord = ac_locs[i+1]
        # y_min = max(y_cord-extend,0)
        # y_max = min(y_cord+extend,511)
        # x_min = max(x_cord-extend,0)
        # x_max = min(x_cord+extend,511)
        x = np.arange(512)
        y = np.arange(512)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-((X-x_cord)**2+(Y-y_cord)**2)/(2*extend**2))
        # response_frame[y_min:y_max,x_min:x_max] = c_response
        gaussian[gaussian<0.2] = 0
        response_frame += c_response*gaussian
    return response_frame
#%%
'''
here are several shuffle functions.
'''
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter

def Shuffler_2D(image,lp_filt = 5):
    F = fft2(image)
    # rows, cols = image.shape
    # crow, ccol = rows // 2, cols // 2
    # radius = r  # Adjust the radius to control the low-pass filter
    # mask = np.ones((rows, cols), np.uint8)
    # cv2.circle(mask, (ccol, crow), radius, 0, -1)
    # F = F * mask
    # Shuffle the phase of the Fourier transform
    phase = np.angle(F)
    amplitude = np.abs(F)
    shuffled_phase = phase + 2 * np.pi * np.random.rand(*phase.shape)
    shuffled_F = amplitude * np.exp(1j * shuffled_phase)
    # Inverse Fourier transform to get the phase-shuffled image
    phase_shuffled_image = np.real(ifft2(shuffled_F))
    phase_shuffled_image = gaussian_filter(phase_shuffled_image,sigma=lp_filt)
    # Normalize the phase-shuffled image
    phase_shuffled_image = (phase_shuffled_image - phase_shuffled_image.min()) / (phase_shuffled_image.max() - phase_shuffled_image.min())
    # get origion min and range
    image_min = image.min()
    range = image.max()-image_min
    phase_shuffled_image = phase_shuffled_image*range+image_min

    return phase_shuffled_image

def Blur_Graph_Recover(blured_graph,ac_locs,extend = 40,weight_min=0.2):

    real_map_1d = blured_graph.flatten()
    # we use linear algebra to calculate the real response.
    weight_matrix = np.zeros(shape = (len(real_map_1d),len(ac_locs.T)),dtype='f8')
    print('Generate Weight Matrix First.')
    for i in tqdm(range(len(ac_locs.T))):
        y_cord,x_cord = ac_locs[i+1]
        x = np.arange(512)
        y = np.arange(512)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-((X-x_cord)**2+(Y-y_cord)**2)/(2*extend**2))
        weight_matrix[:,i] = gaussian.flatten()
    weight_pinv = np.linalg.pinv(weight_matrix,rcond=weight_min)
    cell_weights = np.dot(weight_pinv,real_map_1d)

    return cell_weights

#%% Shuffle graph 100 times, and save corr into a 
N_Shuffle = 100
all_resp = [o0_resp,o22_resp,o45_resp,o67_resp,o90_resp,o112_resp,o135_resp,o157_resp]

all_shuffled_orien_frames = {}
for j in tqdm(range(N_Shuffle)):
    shuffle_resp = []
    shuffle_cell_graph = []
    for i,c_map in enumerate(all_resp):
        c_blur = Graph_Filler(cell_locs=ac_locs,cell_resp=c_map,extend=40,clip=5)
        c_shuffle = Shuffler_2D(image=c_blur,lp_filt=10)
        rec_resp = Blur_Graph_Recover(c_shuffle,ac_locs)
        rec_resp = rec_resp*abs(c_map).max()/abs(rec_resp).max()
        shuffle_resp.append(rec_resp)
        shuffle_cell_graph.append(ac.Generate_Weighted_Cell(rec_resp))
    all_shuffled_orien_frames[j] = shuffle_resp

ot.Save_Variable(r'D:\_Path_For_Figs\240724_Figs_Graph_Shuffle','Shuffle_FuncMaps',all_shuffled_orien_frames)
#%% Test plot, just in case of wired bugs.

plt.clf()
plt.cla()
value_max = 3
value_min = -2
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (12,6),dpi = 180)
cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in range(len(shuffle_cell_graph)):
    # c_map = ac.Generate_Weighted_Cell(all_shuffle_cell_graph[0])
    c_map = shuffle_cell_graph[i]
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//4,i%4],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    # axes[i//4,i%4].set_title(f'PC {i+1}',size = font_size)
fig.tight_layout()


