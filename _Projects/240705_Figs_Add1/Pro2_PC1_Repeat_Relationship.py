'''
This script will try to explain what is PC1, and how it's weight correlate with ensemble.
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
c_spon = np.array(ot.Load_Variable(expt_folder,'Spon_Before.pkl'))
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V2'))
#%%
'''
Analysis 1 : PC1 and Global average are highly correlated.
'''
pcnum = 10
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
avr_all = c_spon.mean(1)
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=2, ncols=1,sharex = True,dpi = 300,figsize = (5,4))
ax[0].plot(spon_coords[4700:5350,0],c = plt.cm.tab10(0))
ax[1].plot(avr_all[4700:5350],c = plt.cm.tab10(1))
r,p = stats.pearsonr(spon_coords[:,0],avr_all)
#%%
'''
Analysis 2 : Ensembles Classified have higher PC1 then unclassified. (Method Quiz cannot be removed, as PC1 also used in SVM), sax-like pattern.
'''
analyzer = Classify_Analyzer(ac = ac,model=spon_model,spon_frame=c_spon,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label

#%% Plot part, we will Plot PC1-3 to show trumpet shape.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
def Plot_Colorized_Oriens(axes,embeddings,labels,pcs=[0,1,2],color_sets = np.zeros(shape = (8,3))):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    orien,orien_id = Select_Frame(embeddings,labels,used_id=list(range(9,17)))
    orien_colors = np.zeros(shape = (len(orien_id),3),dtype='f8')
    for i,c_id in enumerate(orien_id):
        orien_colors[i,:] = color_sets[int(c_id)-9,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 10,c = [0.7,0.7,0.7],alpha = 0.2,lw=0)
    axes.scatter3D(orien[:,0],orien[:,1],orien[:,2],s = 10,c=orien_colors,lw=0)
    return axes

color_setb = np.zeros(shape = (8,3))
fig = plt.figure(figsize = (2,2),dpi = 300)
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)

plt.clf()
plt.cla()
plotted_pcs = [0,1,2]
u = spon_embed
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))
orien_elev = 50
orien_azim = 160
# set axes
ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
ax.axes.set_xlim3d(left=-35, right=50)
ax.axes.set_ylim3d(bottom=-25, top=25)
ax.axes.set_zlim3d(bottom=-40, top=40)
tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                        tmp_planes[0], tmp_planes[1], 
                        tmp_planes[4], tmp_planes[5])
# ax = Plot_Colorized_Oriens(ax,spon_embed,np.zeros(len(spon_embed)),plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,stim_embed,stim_label,plotted_pcs,color_setb)
ax = Plot_Colorized_Oriens(ax,spon_embed,spon_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)
# ax.set_title('Classified Spontaneous in PCA Space',size = 10)
# ax.set_title('Orientation Stimulus in PCA Space',size = 10)
# ax.set_title('Shuffled Spontaneous in PCA Space',size = 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()
#%%
'''
Analysis 3: PC1's weight with Stim-Map Correlation let's see distribution of PC weight and Correlation.
'''
# example loc only.
od_resp = ac.OD_t_graphs['OD'].loc['CohenD',:]
hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD',:]
ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD',:]
red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD',:]
blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD',:]
all_response = [od_resp,hv_resp,ao_resp,red_resp,blue_resp]

networks = ['OD','HV','AO','Red','Blue']
all_corrs = pd.DataFrame(0.0,columns = range(len(c_spon)),index = networks)
# fill it with pearsonr
for i in tqdm(range(len(c_spon))):
    c_response = c_spon[i,:]
    for j,c_net in enumerate(networks):
        c_stim_response = all_response[j]
        c_r,_ = stats.pearsonr(c_response,c_stim_response)
        all_corrs.loc[c_net,i] = abs(c_r)
max_corrs = all_corrs.max(0)
pc1_series = spon_coords[:,1]# other_pcs = spon_coords[:,1:]
plotable = pd.DataFrame([max_corrs,pc1_series],index=['Best Corr','PC1 Weight']).T
#%% Plot Parts

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,5),dpi = 300)
sns.histplot(data = plotable,x = 'PC1 Weight',y = 'Best Corr',ax = ax,bins = 35)



#%%
'''
Analysis 4: PC1's weight correlate with each loc's repeat correlation.
Method 1 - Slide window corr on PC1(Mean std and repeat frame num)
Method 2 - Calculate PC1's peaksum(as globval peaksum), correlate with Repeat Freq.
'''
wp = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig3_PCA_SVM\VARs_From_v4'
orien_similar = ot.Load_Variable(wp,'Orien_Repeat_Similarity.pkl')
orien_freq = ot.Load_Variable(wp,'Orien_Repeat_Freq.pkl')
od_similar = ot.Load_Variable(wp,'OD_Repeat_Similarity.pkl')
od_freq = ot.Load_Variable(wp,'OD_Repeat_Freq.pkl')
hue_similar = ot.Load_Variable(wp,'Hue_Repeat_Similarity.pkl')
hue_freq = ot.Load_Variable(wp,'Hue_Repeat_Freq.pkl')

all_loc_pc1_analysis = pd.DataFrame(columns = ['Loc','PC1_VAR_Ratio','PC1_VAR','PC1_STD_Prop','OD_Freq','Orien_Freq','Color_Freq'])
thres_std = 1
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_orien_freq = orien_freq[orien_freq['Loc']==cloc_name]
    c_orien_freq = c_orien_freq[c_orien_freq['Data_Type']=='Real_Data']['Freq'].iloc[0]
    c_od_freq = od_freq[od_freq['Loc']==cloc_name]
    c_od_freq = c_od_freq[c_od_freq['Data_Type']=='Real_Data']['Freq'].iloc[0]
    c_hue_freq = hue_freq[hue_freq['Loc']==cloc_name]
    c_hue_freq = c_hue_freq[c_hue_freq['Data_Type']=='Real_Data']['Freq'].iloc[0]

    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    pc1_var_ratio = spon_model.explained_variance_ratio_[0]
    pc1_var = spon_model.explained_variance_[0]
    pc1_weight = spon_coords[:,0]
    thres_ratio = (pc1_weight>(pc1_weight.std()*thres_std)).sum()/len(pc1_weight)
    all_loc_pc1_analysis.loc[len(all_loc_pc1_analysis)] = [cloc_name,pc1_var_ratio,pc1_var,thres_ratio,c_od_freq,c_orien_freq,c_hue_freq]
#%% Plot parts
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,4),dpi = 300)
sns.scatterplot(data = all_loc_pc1_analysis,x = 'PC1_VAR',y = 'Orien_Freq',hue = 'Loc',legend=False,ax = ax)
# ax.set_xlim(0,0.5)
# # ax.set_ylim(0,0.1)
# ax.set_ylim(0,0.2)
#%%
'''
Analysis 5, Compare PC1 in V1 and V2
'''
v2_var = []
for i,cloc in enumerate(all_path_dic_v2):
    c_spon = np.array(ot.Load_Variable_v2(cloc,'Spon_Before.pkl'))
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    v2_var.append(spon_model.explained_variance_ratio_[0])

example_loc_v2 = all_path_dic_v2[1]
ac = ot.Load_Variable_v2(example_loc_v2,'Cell_Class.pkl')
c_spon = np.array(ot.Load_Variable(example_loc_v2,'Spon_Before.pkl'))

