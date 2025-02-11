'''
This script will make a supp graph for PC1 info.
It's global response and it's correlate with other network coactivation
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
# all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V2'))


#%%
'''
Fig2 : PC1 and Global average are highly correlated.
'''
pcnum = 10
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
avr_all = c_spon.mean(1)

plt.clf()
plt.cla()
fontsize = 14

fig,ax = plt.subplots(nrows=2, ncols=1,sharex = True,dpi = 300,figsize = (4,4))
ax[0].plot(spon_coords[4700:5350,0],c = plt.cm.tab10(0))
ax[1].plot(avr_all[4700:5350],c = plt.cm.tab10(1))
ax[1].set_xticks(np.array([0,100,200,300,400,500])*1.301)
ax[1].set_xticklabels(np.array([0,100,200,300,400,500]),fontsize = fontsize)
ax[0].set_yticks([-25,0,25,50])
ax[0].set_yticklabels([-25,0,25,50],fontsize = fontsize)
ax[1].set_yticks([-2,0,2])
ax[1].set_yticklabels([-2,0,2],fontsize = fontsize)
fig.subplots_adjust(hspace=0.35)

r,p = stats.pearsonr(spon_coords[:,0],avr_all)
#%% and we will do a stats for all PC1's weight and global corr, return a stats for all pc1 and spon corr.
all_corr = []
for i,cloc in tqdm(enumerate(all_path_dic)):
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    avr_all = c_spon.mean(1)
    r,p = stats.pearsonr(spon_coords[:,0],avr_all)
    all_corr.append(r)

#%% Plot corrs
plotable = pd.DataFrame(all_corr,columns = ['Corr'])

plt.clf()
plt.cla()
fontsize = 14

fig,ax = plt.subplots(nrows=1, ncols=1,sharex = True,dpi = 300,figsize = (2,4))
# sns.barplot(data = plotable,y = 'Corr',ax = ax,width=0.3,capsize = 0.1)
sns.stripplot(data = plotable,ax = ax,y = 'Corr',hue = plotable.index,palette = 'tab10',legend=False,size = 5,linewidth=0)
ax.set_ylim(0.994,1.001)
ax.set_yticks([0.995,1])
ax.set_yticklabels([0.995,1],fontsize = fontsize)
ax.set_ylabel('')

#%% 
'''
Fig 2, we show example of PC1 of given example location.
Reload Example Location plz.
'''
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
pc1_graph = ac.Generate_Weighted_Cell(spon_pcs[0,:])

plt.clf()
plt.cla()
vmax = 0.06
vmin = -0.02
fig,ax = plt.subplots(nrows=1, ncols=1,sharex = True,dpi = 300,figsize = (5,5))

sns.heatmap(pc1_graph,center = 0,xticklabels=False,yticklabels=False,ax = ax,square=True,cmap = 'gist_gray',cbar = False,vmax =vmax,vmin = vmin)

#%% and colorbar 
plt.clf()
plt.cla()
data = [[vmin, vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 10,"shrink": 1,"orientation": "horizontal"},cmap = 'gist_gray')
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([vmin,0,vmax])
g.collections[0].colorbar.set_ticklabels([int(vmin*100),0,int(vmax*100)])
g.collections[0].colorbar.ax.tick_params(labelsize=14)
# g.collections[0].colorbar.aspect(50)
# Create colorbar
# fig.colorbar(ax2.collections[0], ax=ax, orientation='vertical')
plt.show()
#%%
'''
Fig C, Example Location's scatter, trumpet-like shape
'''
analyzer = Classify_Analyzer(ac = ac,model=spon_model,spon_frame=c_spon,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
#%% Plot part

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
# ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
# ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
# ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
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
Fig D, Plot relationship between PC1 weight and best corr.
'''

for i,cloc in enumerate(all_path_dic):
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    od_resp = ac.OD_t_graphs['OD'].loc['CohenD',:]
    hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD',:]
    ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD',:]
    red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD',:]
    blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD',:]
    all_response = [od_resp,hv_resp,ao_resp,red_resp,blue_resp]
    networks = ['OD','HV','AO','Red','Blue']
    cloc_corrs = pd.DataFrame(0.0,columns = range(len(c_spon)),index = networks)
    for k in tqdm(range(len(c_spon))):
        c_response = c_spon[k,:]
        for j,c_net in enumerate(networks):
            c_stim_response = all_response[j]
            c_r,_ = stats.pearsonr(c_response,c_stim_response)
            cloc_corrs.loc[c_net,k] = abs(c_r)
    max_corrs = cloc_corrs.max(0)
    pc1_series = spon_coords[:,0]# other_pcs = spon_coords[:,1:]
    cloc_corr_frame = pd.DataFrame([max_corrs,pc1_series],index=['Best Corr','PC1 Weight']).T
    cloc_corr_frame['Loc'] = cloc.split('\\')[-1]
    cloc_corr_frame['Normed PC1 Weight'] = cloc_corr_frame['PC1 Weight']/abs(cloc_corr_frame['PC1 Weight']).max()
    if i ==0:
        all_corr_frame = copy.deepcopy(cloc_corr_frame)
    else:
        all_corr_frame = pd.concat([all_corr_frame,cloc_corr_frame])
ot.Save_Variable(r'D:\_GoogleDrive_Files\#Figs\#Second_Final_Version-240710\Add2_PC1_Infos','All_PC1_Corr',all_corr_frame)
#%% Plot Parts
        
plt.clf()
plt.cla()
fontsize = 18
# fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,5),dpi = 300)
# sns.histplot(data = all_corr_frame,x = 'Normed PC1 Weight',y = 'Best Corr',ax = ax,bins = 30)
joint = sns.jointplot(data=all_corr_frame, x="Normed PC1 Weight", y="Best Corr", kind="hist",marginal_kws=dict(bins=60),joint_kws=dict(bins=60))
# joint = sns.jointplot(data=all_corr_frame, x="Normed PC1 Weight", y="Best Corr", kind="hist")

joint.ax_joint.set_xlabel('')
joint.ax_joint.set_ylabel('')
joint.ax_joint.set_xticks([-0.8,-0.4,0,0.4,0.8,1.2])
joint.ax_joint.set_xticklabels([-0.8,-0.4,0,0.4,0.8,1.2],fontsize = fontsize)
joint.ax_joint.set_yticks([0,0.2,0.4,0.6,0.8])
joint.ax_joint.set_yticklabels([0,0.2,0.4,0.6,0.8],fontsize = fontsize)
joint.figure.set_figwidth(8)
joint.figure.set_figheight(6)
joint.figure.set_dpi(300)
# r,p = stats.pearsonr(all_corr_frame['Normed PC1 Weight'],all_corr_frame['Best Corr'])
#%% Get colorbar from a single plotted heatmap.
plt.clf()
plt.cla()
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 300)
# fig2, ax2 = plt.subplots()
g = sns.histplot(data=all_corr_frame,x = "Normed PC1 Weight",y = "Best Corr",ax = ax,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "vertical"},bins = 60,cbar = True)
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([0,300])
g.collections[0].colorbar.set_ticklabels([0,300])
g.collections[0].colorbar.ax.tick_params(labelsize=8)
plt.show()