'''
Do temporal PCA, and show the example vector's angle.
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
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl
from Review_Fix_Funcs import *

work_path = r'G:\我的云端硬盘\#Figs\#250211_Revision1\FigS8'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)


#%% Useful Functions
def Angle_Calculate(vec1,vec2): # this function will calculate angle between 2 given vecs.
    norm_v1 = vec1/np.linalg.norm(vec1)
    norm_v2 = vec2/np.linalg.norm(vec2)
    cos = np.dot(norm_v1,norm_v2)
    angle = np.arccos(cos)*180/np.pi
    return angle,cos

def Vector_AVR_Response(vec,pc_comps,spon_frame,ac,centerlize = True): # this will generate average response 
    current_vec_frames = np.dot(pc_comps.T,vec)
    avr_response = np.dot(spon_frame.T,current_vec_frames)
    if centerlize == True:
        avr_response = avr_response-avr_response.mean()
    avr_graph = ac.Generate_Weighted_Cell(avr_response)
    return avr_response,avr_graph

def Vector_Caculator(all_pc_coords,posi_cell_lists,nega_cell_lists = []):
    all_pc_coords = np.array(all_pc_coords)
    posi_coords = np.nan_to_num(all_pc_coords[posi_cell_lists,:].mean(0))
    nega_coords = np.nan_to_num(all_pc_coords[nega_cell_lists,:].mean(0))
    general_corr = posi_coords-nega_coords
    return general_corr



#%% ######################  Example Location PCA #################################
used_pc_num = 20
expt_folder = all_path_dic[2]
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_spon = ot.Load_Variable_v2(expt_folder,'Spon_Before.pkl')
start = c_spon.index[0]
end = c_spon.index[-1]
c_spon = Z_refilter(ac,'1-001',start,end).T

c_ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
print(f'Total {used_pc_num} explained VAR {c_pc_model.explained_variance_ratio_.sum()*100:.1f}%')
# get tuing index of all cells.
ac_oriens = c_ac.all_cell_tunings.loc['Best_Orien',:]
ac_ods = c_ac.all_cell_tunings.loc['OD',:]

# get vecs of OD,HV,AO.
LE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='LE')[0] # this is absolute id.
RE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='RE')[0] # this is absolute id.
OD_vec = Vector_Caculator(c_pc_coords,LE_cells,RE_cells)
Orien0_cells = np.where(ac_oriens == 'Orien0')[0]
Orien90_cells = np.where(ac_oriens == 'Orien90')[0]
HV_vec = Vector_Caculator(c_pc_coords,Orien0_cells,Orien90_cells)
Orien45_cells = np.where(ac_oriens == 'Orien45')[0]
Orien135_cells = np.where(ac_oriens == 'Orien135')[0]
AO_vec = Vector_Caculator(c_pc_coords,Orien45_cells,Orien135_cells)


#%% ########################################################
'''
FIG S5A/B/C - 3 Axis Recovered Response
'''
OD_vec_response,OD_vec_map =  Vector_AVR_Response(vec= OD_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
HV_vec_response,HV_vec_map =  Vector_AVR_Response(vec= HV_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
AO_vec_response,AO_vec_map =  Vector_AVR_Response(vec= AO_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
# real stim funcmaps.
OD_stimmap = c_ac.Generate_Weighted_Cell(c_ac.OD_t_graphs['OD'].loc['CohenD'])
HV_stimmap = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['H-V'].loc['CohenD'])
AO_stimmap = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['A-O'].loc['CohenD'])

rs=[]
ps = []
for i in range(3):
    vec_stim = [c_ac.OD_t_graphs['OD'].loc['CohenD'],c_ac.Orien_t_graphs['H-V'].loc['CohenD'],c_ac.Orien_t_graphs['A-O'].loc['CohenD']][i]
    vec_spon = [OD_vec_response,HV_vec_response,AO_vec_response][i]
    c_r,c_p = stats.pearsonr(vec_stim,vec_spon)
    rs.append(c_r)
    ps.append(c_p)
#%% Plot Stim maps with PC axes
graph_lists = ['HV','AO','OD']
plt.clf()
plt.cla()
value_max = 1.3
value_min = -1
font_size = 12
fig,axes = plt.subplots(nrows=2, ncols=3,figsize = (10,7),dpi = 180)
# cbar_ax = fig.add_axes([.97, .15, .02, .7])
all_pc_axes = [HV_vec_map,AO_vec_map,OD_vec_map]
all_stim_graphs = [HV_stimmap,AO_stimmap,OD_stimmap]

for i,c_map in enumerate(graph_lists):
    c_pc_map = all_pc_axes[i]/all_pc_axes[i].max()
    sns.heatmap(c_pc_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar=False,square=True)
    c_stim_map = all_stim_graphs[i]/all_stim_graphs[i].max()
    sns.heatmap(c_stim_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar=False,square=True)
    # axes[0,i].set_title(c_map,size = font_size)

# axes[0,0].set_ylabel('PCA Functional Axes',rotation=90,size = font_size)
# axes[1,0].set_ylabel('Stimulus Response',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
# fig.tight_layout()
fig.savefig(ot.join(work_path,'S5A_OD_HV_AO_Avr.png'),bbox_inches = 'tight')



#%%
'''
Fig S8B, vector angle stats.
'''

all_angles = pd.DataFrame(columns = ['Axes','Angle','Corr'])
used_pc_num = 20
all_explained_var = np.zeros(len(all_path_dic))
for i,cloc in enumerate(all_path_dic):
    
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable_v2(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(c_ac,'1-001',start,end).T

    c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
    print(f'Total {used_pc_num} explained VAR {c_pc_model.explained_variance_ratio_.sum()*100:.1f}%')
    all_explained_var[i] = c_pc_model.explained_variance_ratio_.sum()
    # get tuing index of all cells.
    ac_oriens = c_ac.all_cell_tunings.loc['Best_Orien',:]
    ac_ods = c_ac.all_cell_tunings.loc['OD',:]
    # get vecs of OD,HV,AO.
    LE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='LE')[0] # this is absolute id.
    RE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='RE')[0] # this is absolute id.
    OD_vec = Vector_Caculator(c_pc_coords,LE_cells,RE_cells)
    Orien0_cells = np.where(ac_oriens == 'Orien0')[0]
    Orien90_cells = np.where(ac_oriens == 'Orien90')[0]
    HV_vec = Vector_Caculator(c_pc_coords,Orien0_cells,Orien90_cells)
    Orien45_cells = np.where(ac_oriens == 'Orien45')[0]
    Orien135_cells = np.where(ac_oriens == 'Orien135')[0]
    AO_vec = Vector_Caculator(c_pc_coords,Orien45_cells,Orien135_cells)

    all_angles.loc[len(all_angles),:] = ['HV-AO',Angle_Calculate(HV_vec,AO_vec)[0],Angle_Calculate(HV_vec,AO_vec)[1]]
    all_angles.loc[len(all_angles),:] = ['OD-AO',Angle_Calculate(OD_vec,AO_vec)[0],Angle_Calculate(OD_vec,AO_vec)[1]]
    all_angles.loc[len(all_angles),:] = ['OD-HV',Angle_Calculate(OD_vec,HV_vec)[0],Angle_Calculate(OD_vec,HV_vec)[1]]

#%% Plot angles.
plt.clf()
# plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 180)
ax.axhline(y = 90,color='gray', linestyle='--')
# sns.boxplot(data = all_angles,y = 'Angle',x = 'Axes',ax = ax,showfliers = 0,legend = True,width=0.5,palette="tab10")
sns.barplot(data = all_angles,y = 'Angle',x = 'Axes',capsize=0.2,palette="tab10",width=0.5)

ax.set_ylim(20,130)
ax.set_yticks([30,60,90,120])
ax.set_yticklabels([30,60,90,120])

# ax.set_title('Functional Axes Included Angle',size = 12,y = 1.05)
# ax.set_xlabel('Axes Pair',weight = 'bold')
# ax.set_ylabel('Angle',weight = 'bold')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels([])
fig.savefig(ot.join(work_path,'S5B_Angels.png'),bbox_inches = 'tight')
plt.show()