'''
This script will generate 2D fourier shuffle of stim map, and we will find that not all maps can be replicated (Only the ensemble ones.)
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
Part 1, generate blured OD graph
'''
ac_locs = ac.Cell_Locs
le_resp = ac.OD_t_graphs['L-0'].loc['CohenD']
re_resp = ac.OD_t_graphs['R-0'].loc['CohenD']
od_resp = ac.OD_t_graphs['OD'].loc['CohenD']

hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD']
ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD']
red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD']
o0_resp = ac.Orien_t_graphs['Orien0-0'].loc['CohenD']
o22_resp = ac.Orien_t_graphs['Orien22.5-0'].loc['CohenD']
o45_resp = ac.Orien_t_graphs['Orien45-0'].loc['CohenD']
o67_resp = ac.Orien_t_graphs['Orien67.5-0'].loc['CohenD']
o90_resp = ac.Orien_t_graphs['Orien90-0'].loc['CohenD']
o112_resp = ac.Orien_t_graphs['Orien112.5-0'].loc['CohenD']
o135_resp = ac.Orien_t_graphs['Orien135-0'].loc['CohenD']
o157_resp = ac.Orien_t_graphs['Orien157.5-0'].loc['CohenD']

def Graph_Filler(cell_locs,cell_resp,extend = 50,clip = 1):
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

# le_frame = Graph_Filler(ac_locs,le_resp,40,5)
# re_frame = Graph_Filler(ac_locs,re_resp,40,5)
# od_frame = Graph_Filler(ac_locs,od_resp,40,5)
o0_frame = Graph_Filler(ac_locs,o0_resp,40,5)
# o22_frame = Graph_Filler(ac_locs,o22_resp,40,5)
# o45_frame = Graph_Filler(ac_locs,o45_resp,40,5)
# o67_frame = Graph_Filler(ac_locs,o67_resp,40,5)
# o90_frame = Graph_Filler(ac_locs,o90_resp,40,5)
# o112_frame = Graph_Filler(ac_locs,o112_resp,40,5)
# o135_frame = Graph_Filler(ac_locs,o135_resp,40,5)
# o157_frame = Graph_Filler(ac_locs,o157_resp,40,5)

#%% Test plot parts
from scipy.ndimage import gaussian_filter
c_graph = o0_frame
filted_graph = c_graph-gaussian_filter(c_graph, sigma=200)
sns.heatmap(filted_graph,cmap='gist_gray',center = 0,xticklabels=False,yticklabels=False,square=True,cbar= None)
#%%
'''
Part 2, get method to shuffle a graph 2D.
'''
from scipy.fft import fft2, ifft2, fftshift
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

shuffled_o0 = Shuffler_2D(o0_frame,lp_filt=10)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(o0_frame, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(shuffled_o0, cmap='gray')
ax2.set_title('Phase-Shuffled Image')
plt.show()
#%% This part will we recover cell from already blured graph.
'''
Part 3, we get shuffled cell graph, and return it back.
'''
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

rec_o0_frame = Blur_Graph_Recover(shuffled_o0,ac_locs)
# rec_o0_frame = Blur_Graph_Recover(o0_frame,ac_locs)
sns.heatmap(ac.Generate_Weighted_Cell(np.clip(rec_o0_frame,-2,2)),cmap = 'gist_gray',center = 0)

#%%
'''
Now let's do it. We will generate all 8 random stim map's response on given location.
Then we will add some random noise to data.
'''
all_resp = [o0_resp,o22_resp,o45_resp,o67_resp,o90_resp,o112_resp,o135_resp,o157_resp]
all_blur = []
all_shuffle = []
all_shuffle_resp = []
all_shuffle_cell_graph = []
for i,c_map in enumerate(all_resp):
    # get blur graph
    c_blur = Graph_Filler(cell_locs=ac_locs,cell_resp=c_map,extend=40,clip=5)
    all_blur.append(c_blur)
    # shuffle blur graph
    c_shuffle = Shuffler_2D(image=c_blur,lp_filt=10)
    all_shuffle.append(c_shuffle)
    # and get cell response.
    rec_resp = Blur_Graph_Recover(c_shuffle,ac_locs)
    rec_resp = rec_resp*abs(c_map).max()/abs(rec_resp).max()
    all_shuffle_resp.append(rec_resp)
    all_shuffle_cell_graph.append(ac.Generate_Weighted_Cell(rec_resp))
# save vars
shuffle_dic = {}
shuffle_dic['Cell_Resp'] = all_resp
shuffle_dic['Cell_Blur'] = all_blur
shuffle_dic['Shuffle_Blur'] = all_shuffle
shuffle_dic['Shuffle_Cell'] = all_shuffle_resp
shuffle_dic['Shuffle_Cell_Map'] = all_shuffle_cell_graph
ot.Save_Variable(r'D:\_GoogleDrive_Files\#Figs\240717_Controls\Dim_Control','Cell_Shuffle',shuffle_dic)
#%% Plot part

plt.clf()
plt.cla()
value_max = 3
value_min = -2
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (12,6),dpi = 180)
cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in range(len(all_shuffle_cell_graph)):
    # c_map = ac.Generate_Weighted_Cell(all_shuffle_cell_graph[0])
    c_map = all_shuffle_cell_graph[i]
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//4,i%4],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    # axes[i//4,i%4].set_title(f'PC {i+1}',size = font_size)

fig.tight_layout()