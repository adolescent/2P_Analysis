'''
This graph compare same graph taken from spon and stimulus stim graph.
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
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.UMAP_Classifier_Analyzer import *

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
# load reducer, if not exist, generate a new one.
reducer = ot.Load_Variable_v2(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')

#%%############################# STEP1, GET COMPARE GRAPHS #########################
analyzer = UMAP_Analyzer(ac = ac,umap_model=reducer,spon_frame=spon_series,od = True,orien = True,color = True,isi = True)
analyzer.Train_SVM_Classifier()
analyzer.Get_Stim_Spon_Compare()
compare_graphs = analyzer.compare_recover
#%%################################# STEP2, GENERATE COMPARE GRAPH###################################
value_max = 2
value_min = -1
font_size = 11
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,5),dpi = 180)

cbar_ax = fig.add_axes([.99, .15, .02, .7])
sns.heatmap(compare_graphs['LE'],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(compare_graphs['RE'],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(compare_graphs['Orien0'],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(compare_graphs['Orien45'],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(compare_graphs['Orien90'],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(compare_graphs['Orien135'],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

axes[0,0].set_title('Left Eye Stimulus                Left Eye Recovered',size = font_size)
axes[1,0].set_title('Right Eye Stimulus              Right Eye Recovered',size = font_size)
axes[0,1].set_title('Orientation0 Stimulus        Orientation0 Recovered',size = font_size)
axes[0,2].set_title('Orientation45 Stimulus      Orientation45 Recovered',size = font_size)
axes[1,1].set_title('Orientation90 Stimulus      Orientation90 Recovered',size = font_size)
axes[1,2].set_title('Orientation135 Stimulus    Orientation135 Recovered',size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
fig.tight_layout()
plt.show()

#%%######################## FIG2c, FRAME SIMILARITY#######################################
# 1. Get all similar and random selection
od_similar,od_similar_rand = analyzer.Similarity_Compare_All(id_lists=list(range(1,9)))
orien_similar,orien_similar_rand = analyzer.Similarity_Compare_All(id_lists=list(range(9,17)))
color_similar,color_similar_rand = analyzer.Similarity_Compare_All(id_lists=list(range(17,23)))

#2. Generate plotable pd frame
distribution_frame = pd.DataFrame(columns = ['Pearson R','Response Pattern','Data'])
for i,c_cond in enumerate(['OD','Orientation','Color']):
    c_data = [od_similar,orien_similar,color_similar][i]
    c_rand = [od_similar_rand,orien_similar_rand,color_similar_rand][i]
    for j,c_frame in enumerate(c_data):
        distribution_frame.loc[len(distribution_frame)] = [c_frame,c_cond,'Data']
    for j,c_frame in enumerate(c_rand):
        distribution_frame.loc[len(distribution_frame)] = [c_frame,c_cond,'Random']
        
#3. Plot violin plot.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3,5),dpi = 180)
axes.axhline(y = 0,color='gray', linestyle='--')
sns.violinplot(data=distribution_frame, x="Response Pattern", y="Pearson R",order = ['OD','Orientation','Color'],hue = 'Data',split=True, inner="quart",ax = axes,dodge= True)
axes.set_title('Spontaneous Repeat Similarity',size = 10)
axes.legend_.remove()
plt.tight_layout()
plt.show()
