'''
This script will do stats on all V1 points. 
Including map repeat freq stats, avr correlation stats,

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

work_path = r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% Define shuffler, change each cell's response curve. This might be changed into another version as phase shuffle.
def Spon_Shuffler(spon_frame):
    shuffled_frame = np.zeros(shape = spon_frame.shape) # output will be an np array, be very careful.
    for i in range(spon_frame.shape[1]):
        c_series = np.array(spon_frame.iloc[:,i])
        np.random.shuffle(c_series)
        shuffled_frame[:,i] = c_series

    return shuffled_frame
# and also define a stim extender, add additional 1 frame after each stim on.
def Stim_Extender(stim_series):
    all_stim_labelv2 = copy.deepcopy(stim_series)
    jump_next = 0
    for i in range(len(all_stim_labelv2)-1):
        if jump_next == 1:
            jump_next = 0
            continue
        if all_stim_labelv2[i]>0 and all_stim_labelv2[i+1] == 0: # extend the end of each stim.
            all_stim_labelv2[i+1] = all_stim_labelv2[i]
            jump_next = 1
    return all_stim_labelv2
#%%######################################################################## Cycle all path and read in all umap model(if not exist, create a umap model.)
# Cycle 1, data preperation. Generate all response map and all umap model.
all_stim_function_map = pd.DataFrame(columns= ['Loc','Response','Stim_ID'])
all_stim_frame_dic = {}
counter = 0
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_stim_frame,c_stim_label = c_ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
    # c_stim_label = Stim_Extender(c_stim_label) # extend 1 frame for discrimination.
    # get tamplate of each location- each stim id.
    all_stim_frame_dic[c_loc.split('\\')[-1]] = (c_stim_frame,c_stim_label)
    for j,c_id in enumerate(list(set(c_stim_label))):
        c_frame_id = np.where((c_stim_label==c_id))[0]
        c_response_map = c_stim_frame.iloc[list(c_frame_id),:].mean(0)
        all_stim_function_map.loc[counter,:] = [c_loc.split('\\')[-1],np.array(c_response_map),c_id]
        counter +=1
    c_reducer = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    if c_reducer == False:
        kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
        c_reducer = umap.UMAP(n_components=3,n_neighbors=20)
        c_reducer.fit(c_stim_frame)
        ot.Save_Variable(c_loc,'All_Stim_UMAP_3D_20comp',c_reducer)
del c_reducer,c_ac
ot.Save_Variable(work_path,'All_Stim_ID_map',all_stim_function_map)
ot.Save_Variable(work_path,'All_Stim_Frame',all_stim_frame_dic)

#%% Cycle 2, get all spon repeat count and spon repeat corr.
all_spon_repeat = pd.DataFrame(columns=['Loc','Raw_Response','Recover_ID','Corr','Data'])
spon_repeat_count = pd.DataFrame(columns=['Loc','Network','Repeat_Count','Repeat_Freq','Data'])
N_shuffle = 10
svm_scores = {}
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

for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_name = c_loc.split('\\')[-1]
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_reducer = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    c_stim_label = all_stim_frame_dic[c_loc_name][1]
    c_stim_embeddings = c_reducer.embedding_
    c_spon_embeddings = c_reducer.transform(c_spon_frame)
    # train the svm classifier and get the score.
    classifier,score = SVM_Classifier(embeddings=c_stim_embeddings,label = c_stim_label)
    predicted_spon_label = SVC_Fit(classifier,data = c_spon_embeddings,thres_prob = 0)
    svm_scores[c_loc_name] = score
    # Next cycle will get repeat event time of 
    eye_repeats = Event_Counter((predicted_spon_label>0)*(predicted_spon_label<9))
    orien_repeats = Event_Counter((predicted_spon_label>8)*(predicted_spon_label<17))
    color_repeats = Event_Counter((predicted_spon_label>16))
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Eye',eye_repeats,eye_repeats*1.301/len(c_spon_frame),'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Orien',orien_repeats,orien_repeats*1.301/len(c_spon_frame),'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Color',color_repeats,color_repeats*1.301/len(c_spon_frame),'Real']
    # Get all recover ID response.
    for j,c_response in enumerate(predicted_spon_label):
        if c_response>0:
            c_spon_frame_single = c_spon_frame.iloc[j,:]
            c_pattern = all_stim_function_map.groupby('Loc').get_group(c_loc_name).groupby('Stim_ID').get_group(c_response)['Response'].iloc[0]
            r,_ = pearsonr(np.array(c_spon_frame_single),c_pattern)
            all_spon_repeat.loc[len(all_spon_repeat),:] = [c_loc_name,c_spon_frame_single,c_response,r,'Real']
            r_rand,_ = pearsonr(c_spon_frame.iloc[np.random.randint(0,len(c_spon_frame))],c_pattern)
            all_spon_repeat.loc[len(all_spon_repeat),:] = [c_loc_name,[],c_response,r_rand,'Random']
            # and calculate a random one.
    ################## Final big part, make shuffle and read all shuffle repeat avr and repeat count.
    for k in range(N_shuffle):
        shuffled_frame = Spon_Shuffler(c_spon_frame)
        embedded_shuffled_frame = c_reducer.transform(shuffled_frame)
        shuffled_label = SVC_Fit(classifier,data = embedded_shuffled_frame,thres_prob = 0)
        eye_repeats_shuffle = Event_Counter((shuffled_label>0)*(shuffled_label<9))
        orien_repeats_shuffle = Event_Counter((shuffled_label>8)*(shuffled_label<17))
        color_repeats_shuffle = Event_Counter((shuffled_label>16))
        spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Eye',eye_repeats_shuffle,eye_repeats_shuffle*1.301/len(c_spon_frame),'Shuffle']
        spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Orien',orien_repeats_shuffle,orien_repeats_shuffle*1.301/len(c_spon_frame),'Shuffle']
        spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_name,'Color',color_repeats_shuffle,color_repeats_shuffle*1.301/len(c_spon_frame),'Shuffle']
        # and get each ID 

ot.Save_Variable(work_path,'All_Spon_Repeat_Stimulus',all_spon_repeat)
ot.Save_Variable(work_path,'All_Spon_Repeat_Network_Coutns',spon_repeat_count)
#%%Fig 2e ##############################################################
# Let's Plot beautiful graphs!
# First is repeat frequency.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,4),dpi = 180)
sns.boxplot(data = spon_repeat_count,x = 'Data',y = 'Repeat_Freq',hue = 'Network',ax = ax)
ax.set_title('Stim-like Ensemble Repeat Frequency')
ax.set_xlabel('Network Type')
ax.set_ylabel('Repeat Frequency(Hz)')
plt.show()

#%%Fig 2f ##############################################################
# This is all repeated frame's corr with model.
# Add label first.
all_spon_repeat['Three_Type_ID'] = ''
all_spon_repeat['Corr'] = all_spon_repeat['Corr'].astype('f8')
for i in tqdm(range(len(all_spon_repeat))):
    c_slice = all_spon_repeat.iloc[i,:]['Recover_ID']
    if c_slice>0 and c_slice<9:
        all_spon_repeat.iloc[i,-1] = 'Single Eye'
    elif c_slice>8 and c_slice<17:
        all_spon_repeat.iloc[i,-1] = 'Orientation'
    elif c_slice>16:
        all_spon_repeat.iloc[i,-1] = 'Color'
# Add full label.
#%%
all_spon_repeat['Full_Type_ID'] = ''
all_spon_repeat['Recover_ID'] = all_spon_repeat['Recover_ID'].astype('i4')
for i in tqdm(range(len(all_spon_repeat))):
    c_slice = all_spon_repeat.loc[i,'Recover_ID']
    if c_slice>0 and c_slice<9 and c_slice%2==1:
        all_spon_repeat.iloc[i,-2] = 'Left Eye'
    elif c_slice>0 and c_slice<9 and c_slice%2==0:
        all_spon_repeat.iloc[i,-2] = 'Right Eye'
    elif c_slice == 9:
        all_spon_repeat.iloc[i,-2] = 'Orientation0'
    elif c_slice == 11:
        all_spon_repeat.iloc[i,-2] = 'Orientation45'
    elif c_slice == 13:
        all_spon_repeat.iloc[i,-2] = 'Orientation90'
    elif c_slice == 15:
        all_spon_repeat.iloc[i,-2] = 'Orientation135'
    elif c_slice == 17:
        all_spon_repeat.iloc[i,-2] = 'Red'
    elif c_slice == 19:
        all_spon_repeat.iloc[i,-2] = 'Green'
    elif c_slice == 21:
        all_spon_repeat.iloc[i,-2] = 'Blue'

#%% Plot repeat similarity
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,5),dpi = 180)
sns.violinplot(data = all_spon_repeat,x = 'Full_Type_ID',y = 'Corr',hue = 'Data',ax = ax, split=True, inner="quart",gap=.1,order=['Left Eye','Right Eye','Orientation0','Orientation45','Orientation90','Orientation135','Red','Green','Blue'])
ax.set_ylim(-0.5,1)
ax.set_title('Ensemble Similarity with Stim-induced Map')
ax.set_xlabel('Network Type')
ax.set_ylabel('Pearson R')
plt.show()
#%%######################################################
# Fig2f-v2 , if we need avr graph.
stimon_frame = all_spon_repeat.groupby('Data').get_group('Real')
corr_frame = pd.DataFrame(columns=['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])
for i,c_loc in tqdm(enumerate(all_path_dic)):

    # get pattern map.
    ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    LE_map = ac.OD_t_graphs['L-0'].loc['A_reponse']
    RE_map = ac.OD_t_graphs['R-0'].loc['A_reponse']
    Orien0_map = ac.Orien_t_graphs['Orien0-0'].loc['A_reponse']
    Orien45_map = ac.Orien_t_graphs['Orien45-0'].loc['A_reponse']
    Orien90_map = ac.Orien_t_graphs['Orien90-0'].loc['A_reponse']
    Orien135_map = ac.Orien_t_graphs['Orien135-0'].loc['A_reponse']
    Red_map = ac.Color_t_graphs['Red-0'].loc['A_reponse']
    # Yellow_map = ac.Color_t_graphs['Yellow-0'].loc['A_reponse']
    Green_map = ac.Color_t_graphs['Green-0'].loc['A_reponse']
    # Cyan_map = ac.Color_t_graphs['Cyan-0'].loc['A_reponse']
    Blue_map = ac.Color_t_graphs['Blue-0'].loc['A_reponse']
    # Purple_map = ac.Color_t_graphs['Purple-0'].loc['A_reponse']
    del ac
    ## get current corr map.
    c_loc_name = c_loc.split('\\')[-1]
    c_loc_repeats = stimon_frame.groupby('Loc').get_group(c_loc_name)
    #LE
    LE_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']>0)*(c_loc_repeats['Recover_ID']<9)*(c_loc_repeats['Recover_ID']%2 == 1)]
    if len(LE_repeat)!=0:
        LE_recover_map = np.zeros(shape = (len(LE_repeat),len(LE_repeat.iloc[0,1])))
        for j in range(len(LE_repeat)):
            LE_recover_map[j,:] = LE_repeat.iloc[j,1]
        LE_recover_map = LE_recover_map.mean(0)
        LE_r,_ = pearsonr(LE_map,LE_recover_map)
    else:
        LE_r = -1
    #RE
    RE_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']>0)*(c_loc_repeats['Recover_ID']<9)*(c_loc_repeats['Recover_ID']%2 == 0)]
    if len(RE_repeat)!=0:
        RE_recover_map = np.zeros(shape = (len(RE_repeat),len(RE_repeat.iloc[0,1])))
        for j in range(len(RE_repeat)):
            RE_recover_map[j,:] = RE_repeat.iloc[j,1]
        RE_recover_map = RE_recover_map.mean(0)
        RE_r,_ = pearsonr(RE_map,RE_recover_map)
    else:
        RE_r = -1
    #Orien0
    Orien0_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==9)]
    if len(Orien0_repeat)!=0:
        Orien0_recover_map = np.zeros(shape = (len(Orien0_repeat),len(Orien0_repeat.iloc[0,1])))
        for j in range(len(Orien0_repeat)):
            Orien0_recover_map[j,:] = Orien0_repeat.iloc[j,1]
        Orien0_recover_map = Orien0_recover_map.mean(0)
        Orien0_r,_ = pearsonr(Orien0_map,Orien0_recover_map)
    else:
        Orien0_r = -1
    #Orien45
    Orien45_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==11)]
    if len(Orien45_repeat)!=0:
        Orien45_recover_map = np.zeros(shape = (len(Orien45_repeat),len(Orien45_repeat.iloc[0,1])))
        for j in range(len(Orien45_repeat)):
            Orien45_recover_map[j,:] = Orien45_repeat.iloc[j,1]
        Orien45_recover_map = Orien45_recover_map.mean(0)
        Orien45_r,_ = pearsonr(Orien45_map,Orien45_recover_map)
    else:
        Orien45_r = -1
    #Orien90
    Orien90_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==13)]
    if len(Orien90_repeat)!=0:
        Orien90_recover_map = np.zeros(shape = (len(Orien90_repeat),len(Orien90_repeat.iloc[0,1])))
        for j in range(len(Orien90_repeat)):
            Orien90_recover_map[j,:] = Orien90_repeat.iloc[j,1]
        Orien90_recover_map = Orien90_recover_map.mean(0)
        Orien90_r,_ = pearsonr(Orien90_map,Orien90_recover_map)
    else:
        Orien90_r = -1
    #Orien135
    Orien135_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==15)]
    if len(Orien135_repeat)!=0:
        Orien135_recover_map = np.zeros(shape = (len(Orien135_repeat),len(Orien135_repeat.iloc[0,1])))
        for j in range(len(Orien135_repeat)):
            Orien135_recover_map[j,:] = Orien135_repeat.iloc[j,1]
        Orien135_recover_map = Orien135_recover_map.mean(0)
        Orien135_r,_ = pearsonr(Orien135_map,Orien135_recover_map)
    else:
        Orien135_r = -1
    #Red
    Red_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==17)]
    if len(Red_repeat)!=0:
        Red_recover_map = np.zeros(shape = (len(Red_repeat),len(Red_repeat.iloc[0,1])))
        for j in range(len(Red_repeat)):
            Red_recover_map[j,:] = Red_repeat.iloc[j,1]
        Red_recover_map = Red_recover_map.mean(0)
        Red_r,_ = pearsonr(Red_map,Red_recover_map)
    else:
        Red_r = -1
    #Green
    Green_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==19)]
    if len(Green_repeat)!=0:
        Green_recover_map = np.zeros(shape = (len(Green_repeat),len(Green_repeat.iloc[0,1])))
        for j in range(len(Green_repeat)):
            Green_recover_map[j,:] = Green_repeat.iloc[j,1]
        Green_recover_map = Green_recover_map.mean(0)
        Green_r,_ = pearsonr(Green_map,Green_recover_map)
    else:
        Green_r = -1
    #Blue
    Blue_repeat = c_loc_repeats[(c_loc_repeats['Recover_ID']==21)]
    if len(Blue_repeat)!=0:
        Blue_recover_map = np.zeros(shape = (len(Blue_repeat),len(Blue_repeat.iloc[0,1])))
        for j in range(len(Blue_repeat)):
            Blue_recover_map[j,:] = Blue_repeat.iloc[j,1]
        Blue_recover_map = Blue_recover_map.mean(0)
        Blue_r,_ = pearsonr(Blue_map,Blue_recover_map)
    else:
        Blue_r = -1
    ### Write into frame.
    corr_frame.loc[len(corr_frame),:] = [LE_r,RE_r,Orien0_r,Orien45_r,Orien90_r,Orien135_r,Red_r,Green_r,Blue_r]
ot.Save_Variable(work_path,'Average_Repeat_Corr',corr_frame)
corr_frame.replace(-1, np.nan, inplace=True)
#%% Plot Fig2f-v2 here.
melted_df = pd.melt(corr_frame,value_name = 'Pearson R',var_name='Type')
melted_df.replace('LE', 'Single Eye', inplace=True)
melted_df.replace('RE', 'Single Eye', inplace=True)
melted_df.replace('Orien0', 'Orientation', inplace=True)
melted_df.replace('Orien45', 'Orientation', inplace=True)
melted_df.replace('Orien90', 'Orientation', inplace=True)
melted_df.replace('Orien135', 'Orientation', inplace=True)
melted_df.replace('Red', 'Color', inplace=True)
melted_df.replace('Yellow', 'Color', inplace=True)
melted_df.replace('Green', 'Color', inplace=True)

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,5),dpi = 180)
# sns.boxplot(data = melted_df,x = 'Type',y = 'Pearson R',ax = ax,order=['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])
sns.boxplot(data = melted_df,x = 'Type',y = 'Pearson R',ax = ax,order=['Single Eye','Orientation','Color'])
ax.set_ylim(0,1)
ax.set_title('Ensemble Similarity with Stim-induced Map')
ax.set_xlabel('Network Type')
ax.set_ylabel('Pearson R')
plt.show()