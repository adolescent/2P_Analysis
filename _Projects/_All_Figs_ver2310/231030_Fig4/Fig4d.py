'''
Fig 4D is the network alteration graph. Plot a pandas and try to visualize them.

'''

#%% Initialization
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
import colorsys

work_path = r'D:\_Path_For_Figs\Fig4_Timecourse_Information'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%%######################  STEP1, GENERATE DATA #######################################
all_spon_prediction = ot.Load_Variable(r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition','All_UMAP_Spon_Prediction.pkl')
all_locs = list(all_spon_prediction.keys())
all_classified_pred_train = {}
all_loc_basic_info = pd.DataFrame(index = range(len(all_locs)),columns=['Loc','OD_Num','Orien_Num','Color_Num','Null_Num','Frame_Num'])
for i,cloc in enumerate(all_locs):
    c_spon_pred = all_spon_prediction[cloc]
    c_predict_train = c_spon_pred[2]
    classified_train = -np.ones(len(c_predict_train))
    # transfer this train into (-1,1,2,3) Train
    od_locs = np.where((c_predict_train>0)*(c_predict_train<9))[0]
    orien_locs = np.where((c_predict_train>8)*(c_predict_train<17))[0]
    color_locs = np.where((c_predict_train>16)*(c_predict_train<23))[0]
    null_locs = np.where((c_predict_train==0))[0]
    classified_train[od_locs] = 1
    classified_train[orien_locs] = 2
    classified_train[color_locs] = 3
    all_classified_pred_train[cloc] = classified_train
    all_loc_basic_info.loc[i,:] = [cloc,len(od_locs),len(orien_locs),len(color_locs),len(null_locs),len(c_predict_train)]
ot.Save_Variable(work_path,'Classified_Prediciton_Train',all_classified_pred_train)
ot.Save_Variable(work_path,'Spon_Predictiom_Basic_Info',all_loc_basic_info)

#%% Get Connection Frame Calculation.
all_connection_info = pd.DataFrame(columns=['Loc','From_Event','To_Event','Count'])
for i,cloc in enumerate(all_locs):
    c_class_train = all_classified_pred_train[cloc]
    # Define 12 connection counter.
    connection_list = []
    for j in range(len(c_class_train)-1):
        from_f = c_class_train[j]
        to_f = c_class_train[j+1]
        connection_list.append((from_f,to_f))
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Null','Null',connection_list.count((-1,-1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Null','OD',connection_list.count((-1,1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Null','Orien',connection_list.count((-1,2))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Null','Color',connection_list.count((-1,3))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'OD','Null',connection_list.count((1,-1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'OD','OD',connection_list.count((1,1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'OD','Orien',connection_list.count((1,2))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'OD','Color',connection_list.count((1,3))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Orien','Null',connection_list.count((2,-1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Orien','OD',connection_list.count((2,1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Orien','Orien',connection_list.count((2,2))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Orien','Color',connection_list.count((2,3))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Color','Null',connection_list.count((3,-1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Color','OD',connection_list.count((3,1))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Color','Orien',connection_list.count((3,2))]
    all_connection_info.loc[len(all_connection_info),:] = [cloc,'Color','Color',connection_list.count((3,3))]
#%% And calculation propotion of each line.
all_connection_info[['Prop_Current','Prop_All']]=0
for i in range(len(all_connection_info)):
    cloc = all_connection_info.loc[i,'Loc']
    c_from_loc = all_connection_info.loc[i,'From_Event']
    c_all_framenum = all_loc_basic_info[all_loc_basic_info['Loc']==cloc].iloc[0,-1]
    if c_from_loc == 'OD':
        c_network_framenum = all_loc_basic_info[all_loc_basic_info['Loc']==cloc].iloc[0,1]
    elif c_from_loc == 'Orien':
        c_network_framenum = all_loc_basic_info[all_loc_basic_info['Loc']==cloc].iloc[0,2]
    elif c_from_loc == 'Color':
        c_network_framenum = all_loc_basic_info[all_loc_basic_info['Loc']==cloc].iloc[0,3]
    elif c_from_loc == 'Null':
        c_network_framenum = all_loc_basic_info[all_loc_basic_info['Loc']==cloc].iloc[0,4]
    all_connection_info.loc[i,'Prop_All'] = all_connection_info.loc[i,'Count']/c_all_framenum
    if c_network_framenum == 0:
        all_connection_info.loc[i,'Prop_Current'] = 0
    else:
        all_connection_info.loc[i,'Prop_Current'] = all_connection_info.loc[i,'Count']/c_network_framenum
ot.Save_Variable(work_path,'All_Connection_Info',all_connection_info)
#%%###################### STEP2,  Plot SINGLE CONNECTION MAP ###########################################
sample_loc = all_connection_info.groupby('Loc').get_group('L76_15A_220812')
averaged_connection = all_connection_info.groupby(['From_Event','To_Event'])['Prop_Current'].mean()
#%% Plot connection graphs.
import matplotlib.pyplot as plt
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize=(6, 6))
null_cen = (0,0)
orien_cen = (0,2)
od_cen = (-1.732,-1)
color_cen = (1.732,-1)
od_orien_vec = (1.732,3)
od_color_vec = (3,0)
orien_color_vec = (1,732,-3)

null_cir = plt.Circle(null_cen, 0.5, color='blue')
orien_cir = plt.Circle(orien_cen, 0.5, color='blue')
# od_cir = plt.Circle(od_cen, 0.2, color='blue', fill=False)
od_cir = plt.Circle(od_cen, 0.5, color='blue')
color_cir = plt.Circle(color_cen, 0.5, color='blue')

ax.add_patch(null_cir)
ax.add_patch(orien_cir)
ax.add_patch(od_cir)
ax.add_patch(color_cir)
ax.set(xlim = (-3,3))
ax.set(ylim = (-3,3))
ax.annotate('Null', xy=null_cen, xytext=null_cen,color = (1,1,1),ha='center',va='center',weight="bold")
ax.annotate('OD', xy=od_cen, xytext=od_cen,color = (1,1,1),ha='center',va='center',weight="bold")
ax.annotate('Orien', xy=orien_cen, xytext=orien_cen,color = (1,1,1),ha='center',va='center',weight="bold")
ax.annotate('Color', xy=color_cen, xytext=color_cen,color = (1,1,1),ha='center',va='center',weight="bold")
def Vector_Calculator(cir1,cir2,radius,shift,start_fix = 0.05,tail_fix = 0.1):
    center_dist = np.sqrt((cir2[0]-cir1[0])**2+(cir2[1]-cir1[1])**2)
    circle_angle = np.arctan2(cir2[1] - cir1[1], cir2[0] - cir1[0])
    shifted_angle = circle_angle+np.radians(shift)
    start_point = (cir1[0]+radius*np.cos(shifted_angle),cir1[1]+radius*np.sin(shifted_angle))
    move_len = center_dist-2*(radius*np.cos(np.radians(shift)))
    vector = (move_len*np.cos(circle_angle),move_len*np.sin(circle_angle))
    return start_point,vector

# calculate color of different connection.
angle_bais = 7
null_orien,null_orien_vec = Vector_Calculator(null_cen,orien_cen,0.6,angle_bais)
ax.arrow(null_orien[0],null_orien[1],null_orien_vec[0],null_orien_vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Null']['Orien']),1))
orien_null,orien_null_vec = Vector_Calculator(orien_cen,null_cen,0.6,angle_bais)
ax.arrow(orien_null[0],orien_null[1],orien_null_vec[0],orien_null_vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Orien']['Null']),1))
od_orien,od_orien_vec = Vector_Calculator(od_cen,orien_cen,0.6,angle_bais)
ax.arrow(od_orien[0],od_orien[1],od_orien_vec[0],od_orien_vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['OD']['Orien']),1))
orien_od,orien_od_vec = Vector_Calculator(orien_cen,od_cen,0.6,angle_bais)
ax.arrow(orien_od[0],orien_od[1],orien_od_vec[0],orien_od_vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Orien']['OD']),1))

# following plot will use simplyfied draw method.
start,vec = Vector_Calculator(orien_cen,color_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Orien']['Color']),1))
start,vec = Vector_Calculator(color_cen,orien_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Color']['Orien']),1))
start,vec = Vector_Calculator(od_cen,color_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['OD']['Color']),1))
start,vec = Vector_Calculator(color_cen,od_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Color']['OD']),1))
start,vec = Vector_Calculator(null_cen,od_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Null']['OD']),1))
start,vec = Vector_Calculator(null_cen,color_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Null']['Color']),1))
start,vec = Vector_Calculator(od_cen,null_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['OD']['Null']),1))
start,vec = Vector_Calculator(color_cen,null_cen,0.6,angle_bais)
ax.arrow(start[0],start[1],vec[0],vec[1],head_width = 0.05,width = 0.01,color =colorsys.hls_to_rgb(0,1-0.5*(averaged_connection['Color']['Null']),1))
# ax.annotate('Arrow', xy=(0.5, 0.5), xytext=(0.2, 0.2))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('Different Network Alteration')

#%%############################ STEP4,SHUFFLE 1000 TIMES ###############################
# Get shuffled data.
N = 1000
all_connection_info_s = pd.DataFrame(columns=['Loc','From_Event','To_Event','Count','Prop_Current','Prop_All'])
all_locs = list(all_spon_prediction.keys())
for i,cloc in enumerate(all_locs):
    c_loc_trains = all_classified_pred_train[cloc]
    c_loc_trains[c_loc_trains==-1]=0
    for j in tqdm(range(N)):
        c_shuffled_train = Shuffle_Multi_Trains(c_loc_trains)
        connection_list = []
        c_od_num = (c_shuffled_train==1).sum()
        c_orien_num = (c_shuffled_train==2).sum()
        c_color_num = (c_shuffled_train==3).sum()
        c_null_num = (c_shuffled_train==0).sum()
        c_all_num = len(c_shuffled_train)
        for k in range(len(c_shuffled_train)-1):
            from_f = c_shuffled_train[k]
            to_f = c_shuffled_train[k+1]
            connection_list.append((from_f,to_f))
        nn_num = connection_list.count((0,0))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Null','Null',nn_num,nn_num/c_null_num,nn_num/c_all_num]
        ne_num = connection_list.count((0,1))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Null','OD',ne_num,ne_num/c_null_num,ne_num/c_all_num]
        no_num = connection_list.count((0,2))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Null','Orien',no_num,no_num/c_null_num,no_num/c_all_num]
        nc_num = connection_list.count((0,3))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Null','Color',nc_num,nc_num/c_null_num,nc_num/c_all_num]
        en_num = connection_list.count((1,0))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'OD','Null',en_num,en_num/c_od_num,en_num/c_all_num]
        ee_num = connection_list.count((1,1))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'OD','OD',ee_num,ee_num/c_od_num,ee_num/c_all_num]
        eo_num = connection_list.count((1,2))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'OD','Orien',eo_num,eo_num/c_od_num,eo_num/c_all_num]
        ec_num = connection_list.count((1,3))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'OD','Color',ec_num,ec_num/c_od_num,ec_num/c_all_num]
        on_num = connection_list.count((2,0))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Orien','Null',on_num,on_num/c_orien_num,on_num/c_orien_num]
        oe_num = connection_list.count((2,1))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Orien','OD',oe_num,oe_num/c_orien_num,oe_num/c_orien_num]
        oo_num = connection_list.count((2,2))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Orien','Orien',oo_num,oo_num/c_orien_num,oo_num/c_orien_num]
        oc_num = connection_list.count((2,3))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Orien','Color',oc_num,oc_num/c_orien_num,oc_num/c_orien_num]
        cn_num = connection_list.count((3,0))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Color','Null',cn_num,cn_num/c_color_num,cn_num/c_color_num]
        ce_num = connection_list.count((3,1))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Color','OD',ce_num,ce_num/c_color_num,ce_num/c_color_num]
        co_num = connection_list.count((3,2))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Color','Orien',co_num,co_num/c_color_num,co_num/c_color_num]
        cc_num = connection_list.count((3,3))
        all_connection_info_s.loc[len(all_connection_info_s),:] = [cloc,'Color','Color',cc_num,cc_num/c_color_num,cc_num/c_color_num]

#%% Average shuffled data to get all avr connection results.
averaged_connection = all_connection_info_s.groupby(['From_Event','To_Event'])['Prop_Current'].mean()