'''
This script will cycle all point, counting number of each repeatance.
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
from Stim_Frame_Align import One_Key_Stim_Align
from scipy.stats import pearsonr
import scipy.stats as stats

all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\_All_Cell_Classes')
all_path = np.delete(all_path,[4,7]) # delete 2 point with not so good OD.

#%% cycle all path,and get point of repeatance of each path.
stats_info = pd.DataFrame(0,columns=['SVM_Correct','Spon_Frames','On_Frames','OD_repeats','Orien_repeats','Color_repeats','Spon_SVM_Train'],index=all_path)
for i,cp in tqdm(enumerate(all_path)):
    kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    spon_series = ot.Load_Variable(cp,'Spon_Before.pkl')
    spon_frame_num = spon_series.shape[0]
    all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
    # train an svm using umap embedded results.
    reducer = umap.UMAP(n_components=3,n_neighbors=20,target_weight=0)
    reducer.fit(all_stim_frame,all_stim_label)
    stim_embeddings = reducer.embedding_
    spon_embeddings = reducer.transform(spon_series)
    classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label)
    svm_series = SVC_Fit(classifier,spon_embeddings,thres_prob = 0)
    on_num = np.sum(svm_series>0)
    od_num = np.sum((svm_series>0)*(svm_series<9))
    orien_num = np.sum((svm_series>8)*(svm_series<17))
    color_num = np.sum(svm_series>16)
    stats_info.loc[cp] = [score,spon_frame_num,on_num,od_num,orien_num,color_num,[svm_series]]
# Do some statistic on datas.
stats_info['On_freq'] = stats_info['On_Frames']*1.301/stats_info['Spon_Frames']
stats_info['OD_freq'] = stats_info['OD_repeats']*1.301/stats_info['Spon_Frames']
stats_info['Orien_freq'] = stats_info['Orien_repeats']*1.301/stats_info['Spon_Frames']
stats_info['Color_freq'] = stats_info['Color_repeats']*1.301/stats_info['Spon_Frames']
# save results.
ot.Save_Variable(r'D:\ZR\_Data_Temp\_Stats','Before_Super_ISI_all',stats_info)
#%% plot bar plot of all propotions.
melted_df = pd.melt(stats_info, value_vars=['On_freq','OD_freq','Orien_freq','Color_freq'], var_name='Type',value_name ='Repeat Frequency (Hz)')
plt.switch_backend('webAgg')
sns.barplot(data = melted_df,y = 'Repeat Frequency (Hz)',x = 'Type')
plt.show()
#%% calculate cosequent event in all spons.
stats_info['ON_event_num'] = 0
stats_info['OD_event_num'] = 0
stats_info['Orien_event_num'] = 0
stats_info['Color_event_num'] = 0
stats_info['ON_event_wid'] = 0
stats_info['OD_event_wid'] = 0
stats_info['Orien_event_wid'] = 0
stats_info['Color_event_wid'] = 0
all_on_len = []
all_od_len = []
all_orien_len = []
all_color_len = []
# cycle all paths.
link_dist = 5
for i,cp in enumerate(all_path):
    c_series = stats_info.loc[cp,'Spon_SVM_Train'][0]
    on_series = c_series>0
    od_series = (c_series>0)*(c_series<9)
    orien_series = (c_series>8)*(c_series<17)
    color_series = (c_series>16)
    # get each event num and event length.
    on_series_raw,on_lens = Label_Event_Cutter(on_series)
    real_ons = []
    for j in range(len(on_series_raw)-1):
        if (on_series_raw[j+1][-1] - on_series_raw[j][0])>link_dist:
            real_ons.append(on_series_raw[j])
    on_event_num = len(real_ons)

    od_series_raw,od_lens = Label_Event_Cutter(od_series)
    real_ods = []
    for j in range(len(od_series_raw)-1):
        if (od_series_raw[j+1][-1] - od_series_raw[j][0])>link_dist:
            real_ods.append(od_series_raw[j])
    od_event_num = len(real_ods)

    orien_series_raw,orien_lens = Label_Event_Cutter(orien_series)
    real_oriens = []
    for j in range(len(orien_series_raw)-1):
        if (orien_series_raw[j+1][-1] - orien_series_raw[j][0])>link_dist:
            real_oriens.append(orien_series_raw[j])
    orien_event_num = len(real_oriens)

    color_series_raw,color_lens = Label_Event_Cutter(color_series)
    real_colors = []
    for j in range(len(color_series_raw)-1):
        if (color_series_raw[j+1][-1] - color_series_raw[j][0])>link_dist:
            real_colors.append(color_series_raw[j])
    color_event_num = len(real_colors)

    # write all results into data frame.
    stats_info.loc[cp,'ON_event_num'] = on_event_num
    stats_info.loc[cp,'ON_event_wid'] = on_lens.mean()
    # stats_info.loc[cp,'ON_event_length'] = [[on_lens]]
    all_on_len.append(on_lens)

    stats_info.loc[cp,'OD_event_num'] = od_event_num
    stats_info.loc[cp,'OD_event_wid'] = od_lens.mean()
    # stats_info.loc[cp,'OD_event_length'] = [[od_lens]]
    all_od_len.append(od_lens)

    stats_info.loc[cp,'Orien_event_num'] = orien_event_num
    stats_info.loc[cp,'Orien_event_wid'] = orien_lens.mean()
    # stats_info.loc[cp,'Orien_event_length'] = [[orien_lens]]
    all_orien_len.append(orien_lens)

    stats_info.loc[cp,'Color_event_num'] = color_event_num
    stats_info.loc[cp,'Color_event_wid'] = color_lens.mean()
    # stats_info.loc[cp,'Color_event_length'] = [[color_lens]]
    all_color_len.append(color_lens)

stats_info['ON_event_length'] = all_on_len
stats_info['OD_event_length'] = all_od_len
stats_info['Orien_event_length'] = all_orien_len
stats_info['Color_event_length'] = all_color_len
# Do some statistic on datas.
stats_info['On_freq_event'] = stats_info['ON_event_num']*1.301/stats_info['Spon_Frames']
stats_info['OD_freq_event'] = stats_info['OD_event_num']*1.301/stats_info['Spon_Frames']
stats_info['Orien_freq_event'] = stats_info['Orien_event_num']*1.301/stats_info['Spon_Frames']
stats_info['Color_freq_event'] = stats_info['Color_event_num']*1.301/stats_info['Spon_Frames']
#%% plot bar plot of all propotions.
melted_df = pd.melt(stats_info, value_vars=['On_freq_event','OD_freq_event','Orien_freq_event','Color_freq_event'], var_name='Type',value_name ='Repeat Frequency (Hz)')
plt.switch_backend('webAgg')
sns.barplot(data = melted_df,y = 'Repeat Frequency (Hz)',x = 'Type')
plt.show()
#%% plot average length
melted_df = pd.melt(stats_info, value_vars=['ON_event_wid','OD_event_wid','Orien_event_wid','Color_event_wid'], var_name='Type',value_name ='Average event width (Frame)')
plt.switch_backend('webAgg')
sns.barplot(data = melted_df,y = 'Average event width (Frame)',x = 'Type')
plt.show()
#%% random shuffle 1000 times of coactivation num.
N = 3000 # shuffle times. Too big will be very slow.
stats_info['Shuffle_Coactivations']=0
stats_info['Shuffle_Coactivation_std']=0
for i,cp in tqdm(enumerate(all_path)):
    spon_frame_num = stats_info.loc[cp,'Spon_Frames']
    c_od_lens = stats_info.loc[cp,'OD_event_length']
    c_orien_lens = stats_info.loc[cp,'Orien_event_length']
    c_color_lens = stats_info.loc[cp,'Color_event_length']
    shuffle_coactivation = np.zeros(N)
    for j in range(N):
        rand_od_series = Random_Series_Generator(spon_frame_num,c_od_lens)
        rand_orien_series = Random_Series_Generator(spon_frame_num,c_orien_lens)
        rand_color_series = Random_Series_Generator(spon_frame_num,c_color_lens)
        rand_series = (rand_od_series+rand_orien_series+rand_color_series)>0
        _,shuffle_events = Label_Event_Cutter(rand_series)
        shuffle_coactivation[j] = shuffle_events.shape[0]
    stats_info.loc[cp,'Shuffle_Coactivations'] = shuffle_coactivation.mean()
    stats_info.loc[cp,'Shuffle_Coactivation_std']=shuffle_coactivation.std()
all_vs_shuffle = stats_info[['ON_event_num','Shuffle_Coactivations','Shuffle_Coactivation_std']]


#%%% Compare recovery graph similarity of each point.
all_recover_similarity = pd.DataFrame(0,columns= ['LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr','Red_Corr','Yellow_Corr','Green_Corr','Cyan_Corr','Blue_Corr','Purple_Corr'],index = all_path)
for i,cp in enumerate(all_path):
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    c_labels = stats_info.loc[cp,'Spon_SVM_Train'][0]
    spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
    # get-0 graphs
    LE = ac.OD_t_graphs['L-0'].loc['t_value']
    RE = ac.OD_t_graphs['R-0'].loc['t_value']
    Orien0 = ac.Orien_t_graphs['Orien0-0'].loc['t_value']
    Orien45 = ac.Orien_t_graphs['Orien45-0'].loc['t_value']
    Orien90 = ac.Orien_t_graphs['Orien90-0'].loc['t_value']
    Orien135 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']
    Red = ac.Color_t_graphs['Red-0'].loc['t_value']
    Yellow = ac.Color_t_graphs['Yellow-0'].loc['t_value']
    Green = ac.Color_t_graphs['Green-0'].loc['t_value']
    Cyan = ac.Color_t_graphs['Cyan-0'].loc['t_value']
    Blue = ac.Color_t_graphs['Blue-0'].loc['t_value']
    Purple = ac.Color_t_graphs['Purple-0'].loc['t_value']
    # calculate each recover maps.
    LE_frames = np.where((c_labels>0)*(c_labels<9)*(c_labels%2 == 1))[0]
    if len(LE_frames)>0:
        LE_recover = spon_frame.iloc[LE_frames,:].mean(0)
        LE_corr,_ = pearsonr(LE,LE_recover)
    else:
        LE_corr = -1
    RE_frames = np.where((c_labels>0)*(c_labels<9)*(c_labels%2 == 0))[0]
    if len(RE_frames)>0:
        RE_recover = spon_frame.iloc[RE_frames,:].mean(0)
        RE_corr,_ = pearsonr(RE,RE_recover)
    else:
        RE_corr = -1
    Orien0_frames = np.where(c_labels == 9)[0]
    if len(Orien0_frames)>0:
        Orien0_recover = spon_frame.iloc[Orien0_frames,:].mean(0)
        Orien0_corr,_ = pearsonr(Orien0,Orien0_recover)
    else:
        Orien0_corr = -1
    Orien45_frames = np.where(c_labels == 11)[0]
    if len(Orien45_frames)>0:
        Orien45_recover = spon_frame.iloc[Orien45_frames,:].mean(0)
        Orien45_corr,_ = pearsonr(Orien45,Orien45_recover)
    else:
        Orien45_corr = -1
    Orien90_frames = np.where(c_labels == 13)[0]
    if len(Orien90_frames)>0:
        Orien90_recover = spon_frame.iloc[Orien90_frames,:].mean(0)
        Orien90_corr,_ = pearsonr(Orien90,Orien90_recover)
    else:
        Orien90_corr = -1
    Orien135_frames = np.where(c_labels == 15)[0]
    if len(Orien135_frames)>0:
        Orien135_recover = spon_frame.iloc[Orien135_frames,:].mean(0)
        Orien135_corr,_ = pearsonr(Orien135,Orien135_recover)
    else:
        Orien135_corr = -1
    Red_frames = np.where(c_labels == 17)[0]
    if len(Red_frames)>0:
        Red_recover = spon_frame.iloc[Red_frames,:].mean(0)
        Red_corr,_ = pearsonr(Red,Red_recover)
    else:
        Red_corr = -1
    Yellow_frames = np.where(c_labels == 18)[0]
    if len(Yellow_frames)>0:
        Yellow_recover = spon_frame.iloc[Yellow_frames,:].mean(0)
        Yellow_corr,_ = pearsonr(Yellow,Yellow_recover)
    else:
        Yellow_corr = -1
    Green_frames = np.where(c_labels == 19)[0]
    if len(Green_frames)>0:
        Green_recover = spon_frame.iloc[Green_frames,:].mean(0)
        Green_corr,_ = pearsonr(Green,Green_recover)
    else:
        Green_corr = -1
    Cyan_frames = np.where(c_labels == 20)[0]
    if len(Cyan_frames)>0:
        Cyan_recover = spon_frame.iloc[Cyan_frames,:].mean(0)
        Cyan_corr,_ = pearsonr(Cyan,Cyan_recover)
    else:
        Cyan_corr = -1
    Blue_frames = np.where(c_labels == 21)[0]
    if len(Blue_frames)>0:
        Blue_recover = spon_frame.iloc[Blue_frames,:].mean(0)
        Blue_corr,_ = pearsonr(Blue,Blue_recover)
    else:
        Blue_corr = -1
    Purple_frames = np.where(c_labels == 22)[0]
    if len(Purple_frames)>0:
        Purple_recover = spon_frame.iloc[Purple_frames,:].mean(0)
        Purple_corr,_ = pearsonr(Purple,Purple_recover)
    else:
        Purple_corr = -1
    all_recover_similarity.iloc[i,:] = [LE_corr,RE_corr,Orien0_corr,Orien45_corr,Orien90_corr,Orien135_corr,Red_corr,Yellow_corr,Green_corr,Cyan_corr,Blue_corr,Purple_corr]
#%% melt and plot recover graph.
melted_frame = pd.melt(all_recover_similarity,value_vars=['LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr','Red_Corr','Yellow_Corr','Green_Corr','Cyan_Corr','Blue_Corr','Purple_Corr'],value_name='recover map r',var_name = 'Map Name')
selected_frame = melted_frame[melted_frame['recover map r']!= -1]
plt.switch_backend('webAgg')
sns.barplot(data = selected_frame,y = 'recover map r',x = 'Map Name',capsize=.2)
plt.show()
#%% calculate average waiting time of all point, all spon on events.
all_wait_time = []
for i,cp in enumerate(all_path):
    c_series = stats_info.loc[cp,'Spon_SVM_Train'][0]
    c_events,_ = Label_Event_Cutter(c_series>0)
    c_waittime = np.zeros(len(c_events)-1)
    for j in range(len(c_events)-1):
        c_waittime[j] = c_events[j+1][0]-c_events[j][0]
    all_wait_time.append(c_waittime)
all_wait_time_str = np.concatenate(all_wait_time)
#%% fit all result with weibull distribution.
param = stats.exponweib.fit(all_wait_time_str,floc = 1)# in sequence (exp1, k1, loc1, lam1)
# param = stats.exponweib.fit(wait_time)
plt.switch_backend('webAgg')
x = np.linspace(0, 200, 200)
pdf_fitted = stats.exponweib.pdf(x, *param)
fig, ax = plt.subplots()
ax.hist(all_wait_time_str, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
ax.plot(x, pdf_fitted, 'r-', label='Fitted')
plt.legend()
plt.show()
stats.kstest(all_wait_time_str,'exponweib',args = param)
#%%QQ plot
params = stats.exponweib.fit(all_wait_time_str,floc = 0.5)
plt.switch_backend('webAgg')
fig, ax = plt.subplots()
stats.probplot(all_wait_time_str, dist=stats.exponweib,sparams = params, plot=ax)
ax.set_title('QQ plot of data vs. Weibull distribution distribution')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
plt.show()
#%% Network alteration with color. Average.
#%% get orientated graphs of all data
clip_thres = 0.002
deci = 20
# make sure we use right labels.
all_edge = []
for i,cp in enumerate(all_path):
    data = stats_info.loc[cp,'Spon_SVM_Train'][0]
    # calculate current edge
    c_framenum = len(data)
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
    for i,ck in enumerate(edge_weights.keys()):
        edge_weights[ck] = edge_weights[ck]/c_framenum
    all_edge.append(edge_weights)
# get all conenctions present at least once.
all_keys = set()
for c_edge in all_edge:
    all_keys.update(c_edge.keys())
# get average connection matrix.
avr_connect_dic = {}
for ck in all_keys:
    c_weight = 0
    for i,c_edge in enumerate(all_edge):
        try:
            c_edge[ck]
        except KeyError:
            c_weight += 0
        else:
            c_weight += c_edge[ck]
    raw_weight = c_weight/len(all_edge)
    processed_weight = round(np.clip(raw_weight,0,clip_thres),deci)
    avr_connect_dic[ck] = processed_weight
# define a vanilla nx graph.
import networkx as nx
plt.switch_backend('webAgg')
G = nx.Graph()
# add all labels as nodes.
# for value in set(data):
#     G.add_node(value)
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
pos[17] = (11.4,-14)
pos[18] = (13,-16)
pos[19] = (11.4,-18)
pos[20] = (10-1.4,-18)
pos[21] = (10-3,-16)
pos[22] = (10-1.4,-14)
# Add edges to the graph and assign weights based on the frequency of each edge
for ck in avr_connect_dic:
    G.add_edge(ck[0], ck[1], weight=avr_connect_dic[ck])
# draww all points
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0]
# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if u != v] # remove all self links
nx.draw_networkx_nodes(G, pos, node_size=700)
edge_colors = [edge_data['weight'] for _, _, edge_data in G.edges(data=True)]
# edge_colors = np.squeeze(np.array(edge_colors)/np.array(edge_colors).max())
G.remove_edges_from(nx.selfloop_edges(G)) # remove self links. have to be here, or bug will raise.
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1,edge_color=edge_colors, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
# edge_labels = nx.get_edge_attributes(G, "weight")
# nx.draw_networkx_edge_labels(G, pos, edge_labels)
ax = plt.gca()
# ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
