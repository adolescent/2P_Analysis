'''

This script will first analyze the prop. of each height event and cluster prop.

Then will do PCA on ON-OFF Frame.

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
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Advanced_Tools import *
from Review_Fix_Funcs import *
from scipy.signal import find_peaks,peak_widths


savepath = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\FigS2'
datapath = r'D:\#Fig_Data\_All_Spon_Data_V1'
all_path_dic = list(ot.Get_Subfolders(datapath))
all_path_dic.pop(4)
all_path_dic.pop(6)

all_peak_info = ot.Load_Variable(savepath,'All_OnOff_Peaks.pkl')
all_on_frames = ot.Load_Variable(savepath,'All_ON_Frames.pkl')

#%% 
"""
Part 1, We calculate each frame's svm class. Try to get the real relationship of each global ensemble's class.

"""
# define frame of all peaks.
all_repeat_info = pd.DataFrame(columns = ['Loc','Thres','Ratio','Type'])

all_thres = np.linspace(0.1,0.8,31)
win_step = 0.1

for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    c_peaks_all = all_peak_info[all_peak_info['Loc']==cloc_name]
    c_class = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')
    for j,c_thres in tqdm(enumerate(all_thres)):
        c_peaks_useful = c_peaks_all[c_peaks_all['Peak_Height']>c_thres]
        c_peaks_useful = c_peaks_useful[c_peaks_useful['Peak_Height']<(c_thres+win_step)]
        peak_num_used = len(c_peaks_useful)
        c_peaks_loc = np.array(c_peaks_useful['Peak_Loc'])# this is the useful class location.

        # count all prop. of repeats.
        c_od = np.array(c_class['OD'])[c_peaks_loc.astype('i4')]>0
        c_orien = np.array(c_class['Orien'])[c_peaks_loc.astype('i4')]>0
        c_color = np.array(c_class['Color'])[c_peaks_loc.astype('i4')]>0
        c_all = c_od+c_orien+c_color

        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_od>0).sum()/peak_num_used,'OD']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_orien>0).sum()/peak_num_used,'Orien']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_color>0).sum()/peak_num_used,'Color']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_all>0).sum()/peak_num_used,'All']

ot.Save_Variable(savepath,'All_Event_SVM_Prop',all_repeat_info)
#%% Plot results above.
plotable = all_repeat_info[all_repeat_info['Type']=='All']
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 300)
sns.lineplot(data = plotable,x = 'Thres',y = 'Ratio',ax = ax)
ax.set_ylim(0,1)
# ax.set_ylabel('Classified Ratio',size = 12)
# ax.set_xlabel('Event Scale Threshold',size = 12)
ax.set_ylabel('')
ax.set_xlabel('')

fig.savefig(ot.join(savepath,'SVM_Classified_Ratio.png'),bbox_inches='tight')

#%%
'''
Part 2, Directly Do PCA on given data frame, and try to compare it with stimulus graph.
'''
example_loc = all_path_dic[2].split('\\')[-1]
example_onframe = all_on_frames[example_loc]
ac = ot.Load_Variable_v2(all_path_dic[2],'Cell_Class.pkl')
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=example_onframe,sample='Frame',pcnum=20)

#%% Plot part
plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=3, ncols=2,figsize = (7,10),dpi = 180)
cbar_ax = fig.add_axes([.98, .35, .015, .2])
vmax = 0.13
vmin = -0.12

for i in range(6):
    c_pc = spon_pcs[i,:]
    c_graph = ac.Generate_Weighted_Cell(c_pc)
    sns.heatmap(c_graph,ax = axes[i//2,i%2],center = 0,vmax = vmax,vmin = vmin,cbar_ax=cbar_ax,xticklabels=False,yticklabels=False,square = True)
    axes[i//2,i%2].set_title(f'PC {i+1}',size = 14)

fig.tight_layout()
