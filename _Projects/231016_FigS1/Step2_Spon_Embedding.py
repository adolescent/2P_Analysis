'''
This script will generate spontaneous embedding on stim-generated space.


'''

#%% Import
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
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl
#%%  ################## Initialization #########################

work_path = r'D:\_Path_For_Figs\S1_PCA_Test'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
cc_path = all_path_dic[2]
# Get all stim and spon frames in cell loc.
ac = ot.Load_Variable_v2(cc_path,'Cell_Class.pkl')
spon_frame = ot.Load_Variable(cc_path,'Spon_Before.pkl')
# get all spon graphs with ISI.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(color = True)
# define function and load pca models in.
model = ot.Load_Variable(work_path,'PCA_Model.pkl')
comps_stim = ot.Load_Variable(work_path,'PC_axis.pkl')
coords_stim = ot.Load_Variable(work_path,'All_Stim_Coordinates.pkl')
stim_vecs = ot.Load_Variable(work_path,'All_Vectors.pkl')
def update(frame,elve = 25):
    ax.view_init(elev=elve, azim=frame)  # Update the view angle for each frame
    return ax,
#%%################# SPON EMBEDDING ########################################
#After stim description,we do spon here. embedding spon onto the stim PC space, and use SVM to classification.
##% 1.train an svc and do reduction.
spon_embeddings = model.transform(spon_frame)
used_dims = coords_stim[:,1:]
classifier,score = SVM_Classifier(used_dims,all_stim_label)
predicted_spon_frame = SVC_Fit(classifier=classifier,data = spon_embeddings[:,1:20],thres_prob=0)

#%%2. Plot spon graph on PC2-4
plt.clf()
plt.cla()
u = spon_embeddings
fig = plt.figure(figsize = (8,6))
elev = 25 # up-down angle
azim = 30 # rotation angle
ax = plt.axes(projection='3d')
ax.grid(False)
ax.view_init(elev=elev, azim=azim)
label = copy.deepcopy(predicted_spon_frame)
label[label>9]=0

sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 3,c = label,cmap = 'turbo')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
fit = stim_vecs['Plane_fits']
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1],5),
                  np.arange(ylim[0], ylim[1],5))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')
from Plot_Tools import Arrow3D
OD_vec = stim_vecs['OD'][1:4]
HV_vec = stim_vecs['HV'][1:4]
AO_vec = stim_vecs['AO'][1:4]
OD_vec = OD_vec/np.linalg.norm(OD_vec)
HV_vec = HV_vec/np.linalg.norm(HV_vec)
AO_vec = AO_vec/np.linalg.norm(AO_vec)
arw1 = Arrow3D([0,-OD_vec[0]*20],[0,-OD_vec[1]*20],[0,-OD_vec[2]*20], arrowstyle="->", color="black", lw = 2, mutation_scale=25)
arw2 = Arrow3D([0,HV_vec[0]*20],[0,HV_vec[1]*20],[0,HV_vec[2]*20], arrowstyle="->", color="red", lw = 2, mutation_scale=25)
arw3 = Arrow3D([0,AO_vec[0]*20],[0,AO_vec[1]*20],[0,AO_vec[2]*20], arrowstyle="->", color="blue", lw = 2, mutation_scale=25)
ax.add_artist(arw1)
ax.add_artist(arw2)
ax.add_artist(arw3)
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
ax.set_title('Recovered Orientation in PC 2-4')
fig.tight_layout()
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')

#%% 3. stats of spon events
def Event_Counter(series): # this function is used to count true list number.
    count = 0
    consecutive_count = 0
    for value in series:
        if value:
            consecutive_count += 1
        else:
            if consecutive_count > 0:
                count += 1
            consecutive_count = 0
    if consecutive_count > 0:
        count += 1
    return count
eye_repeats = Event_Counter((predicted_spon_frame>0)*(predicted_spon_frame<9))
orien_repeats = Event_Counter((predicted_spon_frame>8)*(predicted_spon_frame<17))
color_repeats = Event_Counter((predicted_spon_frame>16))
# def Spon_Shuffler(spon_frame):
#     shuffled_frame = np.zeros(shape = spon_frame.shape) # output will be an np array, be very careful.
#     for i in range(spon_frame.shape[1]):
#         c_series = np.array(spon_frame.iloc[:,i])
#         np.random.shuffle(c_series)
#         shuffled_frame[:,i] = c_series

#     return shuffled_frame
# shuffled_frame = Spon_Shuffler(spon_frame)
# embedded_shuffled_frame = model.transform(shuffled_frame)
# shuffled_label = SVC_Fit(classifier,data = embedded_shuffled_frame[:,:20],thres_prob = 0)
# eye_repeats_shuffle = Event_Counter((shuffled_label>0)*(shuffled_label<9))
# orien_repeats_shuffle = Event_Counter((shuffled_label>8)*(shuffled_label<17))
# color_repeats_shuffle = Event_Counter((shuffled_label>16))
LE_locs = np.where((predicted_spon_frame>0)*(predicted_spon_frame<9)*(predicted_spon_frame%2==1))[0]
LE_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[LE_locs,:].mean(0))
RE_locs = np.where((predicted_spon_frame>0)*(predicted_spon_frame<9)*(predicted_spon_frame%2==0))[0]
RE_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[RE_locs,:].mean(0))
Orien0_locs = np.where(predicted_spon_frame==9)[0]
Orien0_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[Orien0_locs,:].mean(0))
Orien45_locs = np.where(predicted_spon_frame==11)[0]
Orien45_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[Orien45_locs,:].mean(0))
Orien90_locs = np.where(predicted_spon_frame==13)[0]
Orien90_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[Orien90_locs,:].mean(0))
Orien135_locs = np.where(predicted_spon_frame==15)[0]
Orien135_recovered_map = ac.Generate_Weighted_Cell(spon_frame.iloc[Orien135_locs,:].mean(0))


plt.clf()
plt.cla()
value_max = 1.5
value_min = -0.5
font_size = 11

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
sns.heatmap(LE_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(RE_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien0_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien45_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien90_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien135_recovered_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

axes[0,0].set_title('Left Eye Recovered',size = font_size)
axes[1,0].set_title('Right Eye Recovered',size = font_size)
axes[0,1].set_title('Orientation0 Recovered',size = font_size)
axes[0,2].set_title('Orientation45 Recovered',size = font_size)
axes[1,1].set_title('Orientation90 Recovered',size = font_size)
axes[1,2].set_title('Orientation135 Recovered',size = font_size)
fig.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%AXIS LOAD%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This script will plot load of different network.
used_spon_embeddings = spon_embeddings[:,1:4]
OD_load = used_spon_embeddings[:,0]*OD_vec[0]+used_spon_embeddings[:,1]*OD_vec[1]+used_spon_embeddings[:,2]*OD_vec[2]
HV_load = used_spon_embeddings[:,0]*HV_vec[0]+used_spon_embeddings[:,1]*HV_vec[1]+used_spon_embeddings[:,2]*HV_vec[2]
AO_load = used_spon_embeddings[:,0]*AO_vec[0]+used_spon_embeddings[:,1]*AO_vec[1]+used_spon_embeddings[:,2]*AO_vec[2]
norm_vec = stim_vecs['Orien_Normal_PC234']
Orien_Load = np.zeros(len(used_spon_embeddings))
for i in range(len(used_spon_embeddings)):
    c_vec = used_spon_embeddings[i,:]
    c_dist = np.sqrt(np.linalg.norm(c_vec)**2-np.dot(norm_vec,c_vec)**2)
    Orien_Load[i] = c_dist


axis_load = pd.DataFrame(columns=['OD','HV','AO','Orien'])
axis_load['OD'] = OD_load
axis_load['AO'] = AO_load
axis_load['HV'] = HV_load
axis_load['Orien'] = Orien_Load

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12,10),dpi = 180)
pd.plotting.scatter_matrix(axis_load,ax = axes)
fig.suptitle('Network Load',fontsize = 20)
fig.tight_layout()
plt.show()

#%% shuffle spon, and shuffled data have correlation?
def Spon_Shuffler(spon_frame):
    shuffled_frame = np.zeros(shape = spon_frame.shape) # output will be an np array, be very careful.
    for i in range(spon_frame.shape[1]):
        c_series = np.array(spon_frame.iloc[:,i])
        np.random.shuffle(c_series)
        shuffled_frame[:,i] = c_series
    return shuffled_frame

N = 100
OD_AO_corr = np.zeros(N)
OD_HV_corr = np.zeros(N)
HV_AO_corr = np.zeros(N)
for i in tqdm(range(N)):
    shuffled_spon = Spon_Shuffler(spon_frame)
    shuffled_spon_embedding = model.transform(shuffled_spon)
    used_spon_embeddings_shuffle = shuffled_spon_embedding[:,1:4]
    OD_load_s = used_spon_embeddings_shuffle[:,0]*OD_vec[0]+used_spon_embeddings_shuffle[:,1]*OD_vec[1]+used_spon_embeddings_shuffle[:,2]*OD_vec[2]
    HV_load_s = used_spon_embeddings_shuffle[:,0]*HV_vec[0]+used_spon_embeddings_shuffle[:,1]*HV_vec[1]+used_spon_embeddings_shuffle[:,2]*HV_vec[2]
    AO_load_s = used_spon_embeddings_shuffle[:,0]*AO_vec[0]+used_spon_embeddings_shuffle[:,1]*AO_vec[1]+used_spon_embeddings_shuffle[:,2]*AO_vec[2]
    OD_AO_corr[i],_ = stats.pearsonr(OD_load_s,AO_load_s)
    OD_HV_corr[i],_ = stats.pearsonr(OD_load_s,HV_load_s)
    HV_AO_corr[i],_ = stats.pearsonr(HV_load_s,AO_load_s)
    
axis_load_s = pd.DataFrame(columns=['OD','HV','AO'])
axis_load_s['OD'] = OD_load_s
axis_load_s['AO'] = AO_load_s
axis_load_s['HV'] = HV_load_s
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10),dpi = 180)
pd.plotting.scatter_matrix(axis_load_s,ax = axes)
fig.suptitle('Network Load',fontsize = 20)
fig.tight_layout()
plt.show()

#%% Power Analysis
from Analyzer.My_FFT import FFT_Power
# OD_load[abs(OD_load)<3] = 0
# AO_load[abs(AO_load)<3] = 0
# HV_load[abs(HV_load)<3] = 0

# OD_spec = FFT_Power(abs(OD_load),'OD',)
# HV_spec = FFT_Power(abs(HV_load),'HV',)
# AO_spec = FFT_Power(abs(AO_load),'AO',)
# Orien_spec = FFT_Power(Orien_Load,'Orien',)
# All_Power_Spec = pd.DataFrame(columns=['Freq','OD','HV','AO','Orien'])
# All_Power_Spec['Freq'] = np.array(OD_spec.index)
# All_Power_Spec['OD'] = np.array(OD_spec['OD'])
# All_Power_Spec['HV'] = np.array(HV_spec['HV'])
# All_Power_Spec['AO'] = np.array(AO_spec['AO'])  
# All_Power_Spec['Orien'] = np.array(Orien_spec['Orien'])               
# binned_power_sepc = All_Power_Spec.groupby(np.arange(len(All_Power_Spec))//1).mean()
# # Melt data frame and plot.
# plotable_data = pd.melt(binned_power_sepc.iloc[0:30,:], id_vars=['Freq'], value_vars=['OD', 'HV','AO'],var_name='Network',value_name='Power')

# plt.clf()
# plt.cla()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6),dpi = 180)
# sns.lineplot(data = plotable_data,x = 'Freq',y = 'Power',hue = 'Network',ax = ax)
# fig.suptitle('Network Power',fontsize = 20)
# fig.tight_layout()
# plt.show()
import numpy as np
from scipy.signal import csd
frequencies, cross_power_spectrum = csd(abs(OD_load),Orien_Load,fs = 1.301)
plt.semilogy(frequencies, np.abs(cross_power_spectrum))
# plt.plot(frequencies,np.abs(cross_power_spectrum))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD')
plt.title('Cross Power Spectrum of OD and Orientation')
plt.show()

#%% Wait time distribution.
od_repeats = (predicted_spon_frame>0)*(predicted_spon_frame<9)
orien_repeats = (predicted_spon_frame>8)*(predicted_spon_frame<17)
from itertools import groupby
# Find indices of True values
def All_Start_Time(input_series):
    true_indices = np.where(input_series)[0]
    # Find consecutive sequences of True values and their starting indices
    sequences_len = []
    start_index = []
    for k, g in groupby(enumerate(true_indices), lambda ix : ix[0] - ix[1]):
        true_sequence = [x[1] for x in g]
        sequences_len.append(len(true_sequence))
        start_index.append(true_sequence[0])
    # Print the consecutive sequences of True values and their starting indices
    return sequences_len,start_index

od_lens,all_od_locs = All_Start_Time(od_repeats)
orien_lens,all_orien_locs = All_Start_Time(orien_repeats)
all_od_locs = np.array(all_od_locs)
all_orien_locs = np.array(all_orien_locs)
od_nearest_gap = np.zeros(len(all_od_locs))
orien_nearest_gap = np.zeros(len(all_orien_locs))
for i in range(len(od_nearest_gap)):
    c_od_loc = all_od_locs[i]
    od_nearest_gap[i] = abs(all_orien_locs-c_od_loc).min()
for i in range(len(orien_nearest_gap)):
    c_orien_loc = all_orien_locs[i]
    orien_nearest_gap[i] = abs(all_od_locs-c_orien_loc).min()
# get shuffle series 1000 times, make a distribution of random.
from Cell_Class.Advanced_Tools import Random_Series_Generator
series_len = len(od_repeats)
N = 100
all_od_nearest_gap_s = np.zeros(shape = (len(all_od_locs),N))
all_orien_nearest_gap_s = np.zeros(shape = (len(all_orien_locs),N))
for i in tqdm(range(N)):
    c_od_shuffle = Random_Series_Generator(series_len,np.array(od_lens))
    c_orien_shuffle = Random_Series_Generator(series_len,np.array(orien_lens))
    _,all_od_locs_s = All_Start_Time(c_od_shuffle)
    _,all_orien_locs_s = All_Start_Time(c_orien_shuffle)
    for j in range(len(all_od_locs_s)):
        c_od_loc = all_od_locs_s[j]
        all_od_nearest_gap_s[j,i] = abs(all_orien_locs-c_od_loc).min()
    for j in range(len(all_orien_locs_s)):
        c_orien_loc = all_orien_locs_s[j]
        all_orien_nearest_gap_s[j,i] = abs(all_od_locs-c_orien_loc).min()
#%% KS Test
from scipy.stats import kstest
# Assuming you have the population distribution stored in 'population' and the sample in 'sample'
population = all_orien_nearest_gap_s.flatten()
sample = np.array(orien_nearest_gap)

stats.ks_2samp(sample,np.random.choice(population,len(sample)))
