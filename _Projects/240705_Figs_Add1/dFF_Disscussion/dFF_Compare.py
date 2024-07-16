'''
This script will show the difference between dF/F and Z score method.
We want to show that the compromise we made on Z data will cause little trouble.
'''


#%% Load in for example loc.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *



expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
stim_z = ac.Z_Frames[ac.orienrun]
# get dff series
spon_dff = ac.Get_dFF_Frames('1-001',0.1,8500,13852)
spon_dff_avr = ac.Get_dFF_Frames('1-001',0.99,8500,13852)
stim_dff = ac.Get_dFF_Frames(ac.orienrun,0.1)
stim_dff_avr = ac.Get_dFF_Frames(ac.orienrun,0.99)
# make dff series into data frame
spon_dff = pd.DataFrame(spon_dff,columns = range(1,525))
spon_dff_avr = pd.DataFrame(spon_dff_avr,columns = range(1,525))
spon_dff = pd.DataFrame(spon_dff,columns = range(1,525))
stim_dff = pd.DataFrame(stim_dff,columns = range(1,525))
stim_dff_avr = pd.DataFrame(stim_dff_avr,columns = range(1,525))

#%% sort cell ID, for compare convenient
# Sort Orien By Cells actually we sort only by raw data.
rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
for i,cc in enumerate(ac.acn):
    rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
    if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
        rank_index.loc[cc]['Sort_Index']=-1
        rank_index.loc[cc]['Sort_Index2']=0
    else:
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        rank_index.loc[cc]['Sort_Index'] = orien_tunings
        rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
# actually we sort only by raw data.
sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
# and we try to reindex data.
sorted_stim_z = stim_z.T.reindex(sorted_cell_sequence.index).T
sorted_spon_z = spon_series.T.reindex(sorted_cell_sequence.index).T
sorted_stim_dff = stim_dff.T.reindex(sorted_cell_sequence.index).T
sorted_spon_dff = spon_dff.T.reindex(sorted_cell_sequence.index).T
sorted_spon_dff_avr = spon_dff_avr.T.reindex(sorted_cell_sequence.index).T
sorted_stim_dff_avr = stim_dff_avr.T.reindex(sorted_cell_sequence.index).T

#%% 
'''
First, we show the example of dFF and Z Score frame. That will expain why we don't use dFF.
'''
example_dff = sorted_spon_dff.iloc[4700:,:]
example_dff_avr = sorted_spon_dff_avr.iloc[4700:,:]
example_z = sorted_spon_z.iloc[4700:,:]

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,8),dpi = 300,sharex=True)

vmax = 4
vmin = -2
vmax_dff_avr = 0.5
vmin_dff_avr = -0.5
vmax_dff = 2
vmin_dff = -0.5

sns.heatmap((example_dff.T),center = 0,xticklabels=False,yticklabels=False,ax = ax[0],vmax = vmax_dff,vmin = vmin_dff,cbar= False,cmap = 'bwr')
sns.heatmap((example_dff_avr.T),center = 0,xticklabels=False,yticklabels=False,ax = ax[1],vmax = vmax_dff_avr,vmin = vmin_dff_avr,cbar= False,cmap = 'bwr')
sns.heatmap((example_z.T),center = 0,xticklabels=False,yticklabels=False,ax = ax[2],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
#%%
'''
Fig IJK, show another version, we compare dff and z frame.
'''

stim_start_point = 1175
spon_start_point = 5161

stim_recover = stim_z.iloc[stim_start_point:stim_start_point+6].mean(0)
spon_recover = spon_series.iloc[spon_start_point:spon_start_point+6].mean(0)
spon_dff_avr_recover = spon_dff_avr.iloc[spon_start_point:spon_start_point+6].mean(0)
spon_dff_recover = spon_dff.iloc[spon_start_point:spon_start_point+6].mean(0)
stim_dff_recover = stim_dff.iloc[stim_start_point:stim_start_point+6].mean(0)
stim_dff_avr_recover = stim_dff_avr.iloc[stim_start_point:stim_start_point+6].mean(0)

stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)
stim_recover_dff_map = ac.Generate_Weighted_Cell(stim_dff_recover)
stim_recover_dff_avr_map = ac.Generate_Weighted_Cell(stim_dff_avr_recover)
spon_dff_avr_recover_map = ac.Generate_Weighted_Cell(spon_dff_avr_recover)
spon_dff_recover_map = ac.Generate_Weighted_Cell(spon_dff_recover)
#%% Plot part
plt.clf()
plt.cla()
vmax = 0.5
vmin = -0.5
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 300)
sns.heatmap(spon_dff_avr_recover_map,center=0,xticklabels=False,yticklabels=False,ax = ax,vmax = vmax,vmin = vmin,cbar= False,square=True)

savepath = r'D:\_GoogleDrive_Files\#Figs\240705_Add_Infos\dFF_Discussion'
fig.savefig(ot.join(savepath,'Spon_dff_avr_recover.png'))
#%%
'''
Work 3, let's see the result of PCA for dFF data.
'''

frame = copy.deepcopy(spon_dff_avr)

pcnum = 10
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=frame,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_model.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')
#%% Plot PC comps.
plt.clf()
plt.cla()
# value_max = 0.1
# value_min = -0.1
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (12,6),dpi = 300)
cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in tqdm(range(10)):
    c_pc = spon_pcs[i,:]
    c_pc_graph = ac.Generate_Weighted_Cell(c_pc)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar= False,square=True,cmap = 'gist_gray')
    sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],square=True,cmap = 'gist_gray',cbar_ax = cbar_ax)
    # axes[i//5,i%5].set_title(f'PC {i+1}',size = font_size)
fig.tight_layout()
#%%
'''
Work 4, use dff PCA to do SVM. What will we see?
'''
analyzer = Classify_Analyzer(ac = ac,model=spon_model,spon_frame=spon_dff,od = 0,orien = 1,color = 0,isi = True)
analyzer.stim_frame = stim_dff
analyzer.stim_embeddings = analyzer.model.transform(analyzer.stim_frame)
analyzer.Train_SVM_Classifier(C=1)

stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
#%% Plot part
analyzer.Get_Stim_Spon_Compare(od = False,color = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
analyzer.Similarity_Compare_Average(od = False,color = False)
all_corr = analyzer.Avr_Similarity
value_max = 0.3
value_min = -0.3

# Plot Spon and Stim graph seperetly.
plt.clf()
plt.cla()
# cbar_ax = fig.add_axes([.92, .45, .01, .2])
font_size = 14
fig,axes = plt.subplots(nrows=1, ncols=4,figsize = (14,4),dpi = 180)
for i,c_map in enumerate(graph_lists):
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[i],vmax = value_max,vmin = value_min,cbar=False,square=True)

fig.tight_layout()