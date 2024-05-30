'''
This will recover Frame-wise similarity of OD and orientation maps.
Almost the same graph we plot in fig ver4.

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
spon_series = np.array(spon_series)
#%%
'''
Fig S3b-P1, generate svm model and we recover Orientation Frame results.
This is the main result, we might only need this one.

'''

pcnum = 10
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)

analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label

#%% Generate Frame similar maps.
raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')

orien_avr = raw_orien_run.mean(0)
spon_avr = raw_spon_run.mean(0)
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
graph_ids = [9,11,13,15]
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5
# egt all avr stim & spon maps
for i,c_map in tqdm(enumerate(graph_lists)):
    c_spons = raw_spon_run[np.where(spon_label == graph_ids[i])[0],:,:]
    c_spon_avr = c_spons.mean(0)-spon_avr
    c_spon_avr = np.clip(c_spon_avr,(c_spon_avr.mean()-clip_std*c_spon_avr.std()),(c_spon_avr.mean()+clip_std*c_spon_avr.std()))
    c_stims = raw_orien_run[np.where(stim_label == graph_ids[i])[0],:,:]
    c_stim_avr = c_stims.mean(0)-orien_avr
    c_stim_avr = np.clip(c_stim_avr,(c_stim_avr.mean()-clip_std*c_stim_avr.std()),(c_stim_avr.mean()+clip_std*c_stim_avr.std()))
    all_stim_maps[c_map] = c_stim_avr 
    all_spon_recover_maps[c_map] = c_spon_avr
#%% Plot Frame Similar maps.
frame_corrs = []
plt.clf()
plt.cla()
value_max = 4
value_min = -4
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])

for i,c_map in enumerate(graph_lists):
    plotable_c_stim = all_stim_maps[c_map]
    plotable_c_stim = plotable_c_stim/plotable_c_stim.std()
    sns.heatmap(plotable_c_stim,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

    plotable_c_spon = all_spon_recover_maps[c_map]
    plotable_c_spon = plotable_c_spon/plotable_c_spon.std()
    c_corr,_ = stats.pearsonr(all_spon_recover_maps[c_map][20:492,20:492].flatten(),all_stim_maps[c_map][20:492,20:492].flatten())
    frame_corrs.append(c_corr)
    sns.heatmap(plotable_c_spon,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[1,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[0,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

#%%
'''
Fig S3b-P2, generate svm model and we recover OD Frame results.
This is the sub result.
'''

pcnum = 10
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)

analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 1,orien = 0,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label

#%% Generate OD Frame recover maps.
raw_od_run = ot.Load_Variable(f'{wp}\\OD_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')
orien_avr = raw_od_run.mean(0)
spon_avr = raw_spon_run.mean(0)

spon_label_od = np.zeros(len(spon_label))
for i in range(len(spon_label)):
    if (spon_label[i]>0) and (spon_label[i]%2 == 1):
        spon_label_od[i] = 1
    elif (spon_label[i]>0):
        spon_label_od[i] = 2
    else:
        spon_label_od[i] = 0
stim_label_od = np.zeros(len(stim_label))
for i in range(len(stim_label)):
    if (stim_label[i]>0) and (stim_label[i]%2 == 1):
        stim_label_od[i] = 1
    elif (stim_label[i]>0):
        stim_label_od[i] = 2
    else:
        stim_label_od[i] = 0

graph_lists = ['LE','RE']
graph_ids = [1,2]
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5
# egt all avr stim & spon maps
for i,c_map in tqdm(enumerate(graph_lists)):
    c_spons = raw_spon_run[np.where(spon_label_od == graph_ids[i])[0],:,:]
    c_spon_avr = c_spons.mean(0)-spon_avr
    c_spon_avr = np.clip(c_spon_avr,(c_spon_avr.mean()-clip_std*c_spon_avr.std()),(c_spon_avr.mean()+clip_std*c_spon_avr.std()))
    c_stims = raw_od_run[np.where(stim_label_od == graph_ids[i])[0],:,:]
    c_stim_avr = c_stims.mean(0)-orien_avr
    c_stim_avr = np.clip(c_stim_avr,(c_stim_avr.mean()-clip_std*c_stim_avr.std()),(c_stim_avr.mean()+clip_std*c_stim_avr.std()))
    all_stim_maps[c_map] = c_stim_avr 
    all_spon_recover_maps[c_map] = c_spon_avr
#%% Plot Graph Compare Maps
frame_corrs = []
plt.clf()
plt.cla()
value_max = 4
value_min = -4
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=2,figsize = (6,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
for i,c_map in enumerate(graph_lists):
    plotable_c_stim = all_stim_maps[c_map]
    plotable_c_stim = plotable_c_stim/plotable_c_stim.std()
    sns.heatmap(plotable_c_stim,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    plotable_c_spon = all_spon_recover_maps[c_map]
    plotable_c_spon = plotable_c_spon/plotable_c_spon.std()
    c_corr,_ = stats.pearsonr(all_spon_recover_maps[c_map][20:492,20:492].flatten(),all_stim_maps[c_map][20:492,20:492].flatten())
    frame_corrs.append(c_corr)
    sns.heatmap(plotable_c_spon,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[1,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[0,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

#%%
'''
Fig S3b-P3, generate svm model and we recover Color Frame results.
This is the sub result.
'''
pcnum = 10
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)

analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 0,color = 1,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label

#%% Generate Color Frame Maps
raw_hue_run = ot.Load_Variable(f'{wp}\\Color_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')
hue_avr = raw_hue_run.mean(0)
spon_avr = raw_spon_run.mean(0)

graph_lists = ['Red','Green','Blue']
graph_ids = [17,19,21]
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5
# egt all avr stim & spon maps
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5

# get all avr stim & spon maps
for i,c_map in tqdm(enumerate(graph_lists)):
    c_spons = raw_spon_run[np.where(spon_label == graph_ids[i])[0],:,:]
    c_spon_avr = c_spons.mean(0)-spon_avr
    c_spon_avr = np.clip(c_spon_avr,(c_spon_avr.mean()-clip_std*c_spon_avr.std()),(c_spon_avr.mean()+clip_std*c_spon_avr.std()))
    c_stims = raw_hue_run[np.where(stim_label == graph_ids[i])[0],:,:]
    c_stim_avr = c_stims.mean(0)-hue_avr
    c_stim_avr = np.clip(c_stim_avr,(c_stim_avr.mean()-clip_std*c_stim_avr.std()),(c_stim_avr.mean()+clip_std*c_stim_avr.std()))
    all_stim_maps[c_map] = c_stim_avr 
    all_spon_recover_maps[c_map] = c_spon_avr

#%% Plot Graph Compare Maps
frame_corrs = []
plt.clf()
plt.cla()
value_max = 5
value_min = -4
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=3,figsize = (9,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
for i,c_map in enumerate(graph_lists):
    plotable_c_stim = all_stim_maps[c_map]
    plotable_c_stim = plotable_c_stim/plotable_c_stim.std()
    sns.heatmap(plotable_c_stim,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

    plotable_c_spon = all_spon_recover_maps[c_map]
    plotable_c_spon = plotable_c_spon/plotable_c_spon.std()
    c_corr,_ = stats.pearsonr(all_spon_recover_maps[c_map][20:492,20:492].flatten(),all_stim_maps[c_map][20:492,20:492].flatten())
    frame_corrs.append(c_corr)
    sns.heatmap(plotable_c_spon,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[1,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[0,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()