'''
Repeat all procedure, but using only supervised umap as labeler.
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from scipy.stats import pearsonr
import scipy.stats as stats

wp = r'D:\ZR\_Data_Temp\_All_Cell_Classes\220420_L91'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(isi = True)
#%% train an supervised svm.
reducer_sup = umap.UMAP(n_components= 3,n_neighbors=20,target_weight=0)
reducer_sup.fit(all_stim_frame,all_stim_label)
u = reducer_sup.embedding_
spon_embedding = reducer_sup.transform(spon_frame)
#plot 3d
plt.switch_backend('webAgg')
fig,ax =Plot_3D_With_Labels(u,all_stim_label)
# ax.scatter3D(spon_embedding[:,0],spon_embedding[:,1],spon_embedding[:,2],c = 'r',s = 3)
plt.show()
# Save_3D_Gif(ax,fig)
#%% and train svm
classifier,score = SVM_Classifier(u,all_stim_label)
spon_svm_label = SVC_Fit(classifier,spon_embedding,0)
plt.switch_backend('webAgg')
fig,ax =Plot_3D_With_Labels(spon_embedding,spon_svm_label)
plt.show()
event,length = Label_Event_Cutter(spon_svm_label>0)
#%% Shuffle to random combine OD/Orien networks.
spike_series_od = (spon_svm_label>0)*(spon_svm_label<9)
spike_series_orien = (spon_svm_label>0)*(spon_svm_label>8)
od_events,od_event_len = Label_Event_Cutter(spike_series_od)
orien_events,orien_event_len = Label_Event_Cutter(spike_series_orien)
all_events,all_len = Label_Event_Cutter(spon_svm_label>0)
# shuffle.
N = 10000# it's a little slow, so don't run 100000 every time.
shuffle_shape = pd.DataFrame(0,index = range(N),columns=['Coactivation','mean_length','std'])
for i in tqdm(range(N)): # cycle shuffle
    # generate random OD series
    rand_od_series = Random_Series_Generator(11554,od_event_len)
    rand_orien_series = Random_Series_Generator(11554,orien_event_len)
    combined_series = (rand_od_series+rand_orien_series)>0
    rand_event,rand_len = Label_Event_Cutter(combined_series)
    shuffle_shape.loc[i,:] = [len(rand_event),rand_len.mean(),rand_len.std()]

# plot distribution of random combine.
plt.switch_backend('webAgg')
ax = sns.histplot(data = shuffle_shape,x = 'Coactivation')
plt.show()
#%% compare whether time distribution obey webb distribution.
all_spike_series = (spon_svm_label>0)
all_event,all_length = Label_Event_Cutter(all_spike_series)
wait_time = np.zeros(len(all_event)-1)
for i in range(1,len(all_event)):
    before_time = all_event[i-1][0]
    after_time = all_event[i][0]
    c_wait_time = after_time-before_time
    wait_time[i-1] = c_wait_time

param = stats.exponweib.fit(wait_time,floc = 0.5)# in sequence (exp1, k1, loc1, lam1)
# param = stats.exponweib.fit(wait_time)
plt.switch_backend('webAgg')
x = np.linspace(0, 300, 300)
pdf_fitted = stats.exponweib.pdf(x, *param)
fig, ax = plt.subplots()
ax.hist(wait_time, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
ax.plot(x, pdf_fitted, 'r-', label='Fitted')
plt.legend()
plt.show()
stats.kstest(wait_time,'exponweib',args = param)
#%% QQ Plot
params = stats.exponweib.fit(wait_time,floc = 0.5)
plt.switch_backend('webAgg')
fig, ax = plt.subplots()
stats.probplot(wait_time, dist=stats.exponweib,sparams = params, plot=ax)
ax.set_title('QQ plot of data vs. Weibull distribution distribution')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
plt.show()
#%% color as ensemble scale.
svc_on_frames = all_labels>0
on_thres = 2
cell_counts = np.array((spon_frame>on_thres).sum(1))
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
fig.tight_layout()
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.set_position([0.1, 0.2, 0.8, 0.7])
ax.scatter3D(spon_embedding[:,0],spon_embedding[:,1],spon_embedding[:,2],s = 3,c = (cell_counts>5)*(cell_counts<10),cmap = 'plasma')
# Save_3D_Gif(ax,fig)
plt.show()
#%% get ensemble repeat ratio by cell count thres.
thres = np.linspace(4,200,49)
ensemble_counts = np.zeros(thres.shape)
ensemble_repeats = np.zeros(thres.shape)
ensemble_peaklen = np.zeros(thres.shape)
ensemble_repeat_ratio = np.zeros(thres.shape)
for i,c_thres in tqdm(enumerate(thres)):
    event_trains = cell_counts>c_thres
    cutted_events,cutted_len = Label_Event_Cutter(event_trains)
    ensemble_counts[i] = len(cutted_events)
    ensemble_peaklen[i] = cutted_len.mean()
    c_repeat = 0
    for j,c_event in enumerate(cutted_events):
        if svc_on_frames[c_event].sum()>0: # for any event, one frame repeat will be repeat.
            c_repeat += 1
    ensemble_repeats[i] = c_repeat
    ensemble_repeat_ratio[i] = c_repeat/len(cutted_events)

plt.switch_backend('webAgg')
plt.plot(thres,ensemble_repeat_ratio)
plt.show()


#%% get all biggest peak of all events.
small_events,small_event_count =  Label_Event_Cutter(cell_counts>20)
biggest_events = np.zeros(small_event_count.shape)
for i,c_event in enumerate(small_events):
    biggest_events[i] = cell_counts[c_event].max()
plt.switch_backend('webAgg')
plt.hist(biggest_events,bins = 50)
plt.show()
prop = np.sum(biggest_events>50)/np.sum(biggest_events>0)
# and all peak of svc frames.
svm_locs = np.where(all_spike_series)[0]
svm_scales = cell_counts[svm_locs]
#%% check whether these small peak are in svm labels.
repeats = np.zeros(np.sum(biggest_events<50))
count = 0
for i,c_repeat in enumerate(small_events):
    if biggest_events[i]<50: # means it's a small event
        c_sequence = small_events[i]
        if all_spike_series[c_sequence].sum()>0:
            repeats[count] = 1
            count+=1
#%% get orientated graphs of all data
import networkx as nx
# make sure we use right labels.
data = spon_svm_label
# define a vanilla nx graph.
G = nx.Graph()
# add all labels as nodes.
for value in set(data):
    G.add_node(value)
pos = nx.spring_layout(G) 
# set position of each point manually.
pos[1]=(0,2)
pos[2]=(20,2)
pos[3]=(-2,0)
pos[4]=(18,0)
pos[5]=(0,-2)
pos[6]=(22,0)
pos[7]=(2,0)
pos[8]=(20,-2)
pos[9]=(10,2)
pos[10]=(10-1.4,1.4)
pos[11]=(8,0)
pos[12]=(10-1.4,-1.4)
pos[13]=(10,-2)
pos[14]=(11.4,-1.4)
pos[15]=(12,0)
pos[16]=(11.4,1.4)
pos[0]=(10,-8)# small adjustment of null
# Add edges to the graph and assign weights based on the frequency of each edge
edge_weights = {}
for i in range(len(data)-1):
    edge = (data[i], data[i+1])
    edge_rev = (data[i+1], data[i])
    if (edge in edge_weights):# make both side conenction even.
        edge_weights[edge] += 1
    elif (edge_rev in edge_weights):
        edge_weights[edge_rev] += 1
    else:
        edge_weights[edge] = 1
    G.add_edge(*edge)
# clip edge_weights to avoid null of all results.
max_weight = 200
for i,c_link in enumerate(list(edge_weights.keys())):
    edge_weights[c_link] = min(edge_weights[c_link],max_weight)
# Set the weights of the edges
for edge, weight in edge_weights.items():
    G[edge[0]][edge[1]]['weight'] = weight

# Draw the graph with edge weights as edge colors
edge_colors = [edge_data['weight'] for _, _, edge_data in G.edges(data=True)]

plt.switch_backend('webAgg')
nx.draw_networkx(G, pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
plt.show()