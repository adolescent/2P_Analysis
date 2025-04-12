'''
Here are several new results, not shown in precvious graphs.
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
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings


warnings.filterwarnings("ignore")

expt_folder = r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1\L76_18M_220902'
# savepath = r'D:\_GoogleDrive_Files\#卒论\Figs'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
spon_before = Z_refilter(ac,'1-001',start,end).T
spon_after = Z_refilter(ac,'1-003').T
#%% Plot before and after
plt.cla()
plt.clf()
fontsize = 14
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,4),dpi = 240)
sns.heatmap(spon_before.T,center=0,cmap='bwr',vmax = 3,vmin = -2,cbar=False,xticklabels=False,yticklabels=False)
fps = 1.301
axes.set_xticks([0,30*60*fps,60*60*fps,90*60*fps,120*60*fps,150*60*fps,180*60*fps])
axes.set_xticklabels([0,30,60,90,120,150,180],fontsize = fontsize)

#%% and plot a slide window fft.
avr_series = spon_before.mean(1)
def FFT_Spectrum_Slide(series,win_size,win_step,fps,ticks=0.01,normalize = False):

    winnum = (len(series)-win_size)//win_step+1
    for i in tqdm(range(winnum)):
        c_series = series[i*win_step:i*win_step+win_size]
        freq_ticks,c_spectrum,_,_,c_power = FFT_Spectrum(c_series,fps,ticks)
        # initialize pd frame.
        if i == 0:
            power_spectrum = pd.DataFrame(0.0,columns = range(winnum),index = freq_ticks.round(3))
        if normalize == True:
            power_spectrum.loc[:,i] = c_spectrum
        else:
            power_spectrum.loc[:,i] = c_spectrum*c_power
    return power_spectrum

slide_spectrum = FFT_Spectrum_Slide(series=avr_series,win_size=int(300*1.301),win_step=int(60*1.301),fps=1.301,ticks=0.01,normalize=False) 
# Plot slide window power.

plt.cla()
plt.clf()
fontsize = 14
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,4),dpi = 240)
sns.heatmap(slide_spectrum,center=0,cmap='bwr',vmax = 5,vmin = 0,cbar=False,xticklabels=False,yticklabels=False,ax = axes)
axes.set_xticks([0,30,60,90,120,150,180])
axes.set_xticklabels([0,30,60,90,120,150,180],fontsize = fontsize)
axes.set_yticks([0,20,40,60])
axes.set_yticklabels([0,0.2,0.4,0.6],fontsize = fontsize)

#%%
'''
Hierachical Cluster of given data.

'''


from scipy.spatial.distance import pdist,squareform

distance_metric = 'euclidean'
# cityblock return L1 norm of data. Different matrix return some different results.
# Multiple options: 'euclidean', 'correlation', 'cityblock','chebyshev',.....

distance_matrix = pdist(spon_before.T , metric=distance_metric)
distance_matrix.shape

#%% distance matrix
square_matrix = squareform(distance_matrix) 
sns.heatmap(square_matrix)

# you can also use this function to transform distance matrix into dense 1D corr info.
recovered_matrix = squareform(square_matrix)
# test whether recovered matrix is the same as original
print((recovered_matrix == distance_matrix).min())

#%% linkage
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
Z = linkage(distance_matrix, method='ward',metric=distance_metric )

print(Z.shape)
print(Z[0])
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,6),dpi = 180)
a = dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=25000,                   #Steps of cluster to show. set very big will show whole dentrogram.
    show_leaf_counts=True,
    leaf_rotation=90, # rotation angle of leaf 
    leaf_font_size=6, # fontsize of leaf
    ax = ax,
    orientation= 'top' # Orientation of cluster going on. top as default.
)
ax.set_title('Dendrogram')
ax.set_xlabel('Samples')
ax.set_ylabel('Distance')
ax.set_xticks([]) # Mute x label if you want full graph.
#%% get cluster.
import matplotlib.cm as cm
n_clusters = 8
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
# cutoff_distance =2
# clusters = fcluster(Z, t=cutoff_distance, criterion='distance')
# print(clusters) # cluster is a list of id, indicating the cluster of each pix.
n_clust = len(set(np.array(clusters)))
#%recover cluster 
a = ac.Generate_Weighted_Cell(np.array(clusters)/n_clust)
sns.heatmap(a,cmap='rainbow',square=True,xticklabels=False,yticklabels=False,cbar=False)

#%% Plot normal orientation and color map.
