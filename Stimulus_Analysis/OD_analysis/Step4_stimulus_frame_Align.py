# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:32:46 2019

@author: ZR
"""
#%%
import neo
import numpy as np

# create a reader
reader = neo.io.Spike2IO(filename=r'G:\ZR\data_processing\190412_L74_LM\190412_L74_stimuli\Run03_2P_manual_OD8\3.smr')
# read the block
data = reader.read( lazy=False)[0]
#%%读取stim的id顺序
f = open(r'G:\ZR\data_processing\190412_L74_LM\190412_L74_stimuli\Run03_2P_manual_OD8\Runxx_OD8_manual_27.5689.txt','r')
stim_id = f.read().split()
f.close()
pre_stim_frame = 2#这里指的是方波开始刺激没放，需要删除的帧数。
after_stim_drift = 0#这里指的是锯齿波消失之后，再计算几帧属于其中。
#%%读取两个spike2的数据序列，并保存为字典。
Stimuli = {}
twoP = {}#将两个序列存入字典，包括了时间、信号序列、采样频率。
for i, asig in enumerate(data.segments[0].analogsignals):
        # 提取时间，并化为秒作单位。
    times = asig.times.rescale('s').magnitude
    # Determine channel name, without leading b' 
    ch = str(asig.annotations['channel_names']).split(sep="'")[1]
    # Extract sampling frequency
    fs = float(asig.sampling_rate)
    # Assign sampling times, sampling frequency and data to correct dictionary
    if ch == 'Stimuli':
        Stimuli['times'] = times
        Stimuli['signal'] = np.array(asig)
        Stimuli['fs'] = fs
    elif ch == '2P':
        twoP['times'] = times
        twoP['signal'] = np.array(asig)
        twoP['fs'] = fs

#%%确定stimOn的序列。
all_stim_time = [x for x in range(len(Stimuli['signal'])) if Stimuli['signal'][x] > 2] # 一次获得所有刺激态的坐标,认为大于2的是刺激ON
#接下来将以上序列进行分割，根据时间连续性分割成了n个序列。
import more_itertools as mit
stim_lists = [list(group) for group in mit.consecutive_groups(all_stim_time)]
#定义一个异常，以免出现计算错误
if len(stim_lists)!=len(stim_id):
    raise Exception('Stim sequence not match, please recheck stim ids.\n')
#将stim_id与stim_lists相对应，并将刺激id对应的时间归到一个字典文件里。
Stim_ID_Time = {}#定义空字典，准备加入每个id和它对应的时间序列。
stim_set = list(set(stim_id))
stim_set.sort()
#遍历每一个刺激id，得到每个刺激id对应的时间。
for i in range(0,len(stim_set)):
    temp_index = [j for j in range(len(stim_id)) if stim_id[j] == stim_set[i]]#这里得到的是所有stim_id为i的序列。
    temp_stim_time = []#定义空列表，作为每一个刺激id的时间暂存器
    for k in range(0,len(temp_index)):
        temp_stim_time.extend(stim_lists[temp_index[k]])
    Stim_ID_Time[stim_set[i]] = temp_stim_time
#到此为止，我们得到了每个刺激id播放时对应的时间。
#%% 接下来确定每一帧的序列。需要找到每一帧的时间。
frame_time = []#定义空列表，记录每一帧播放时在spike2上记录的时间。
i = 20
while i<(len(times)-20):
    if (twoP['signal'][i-20]<0 and twoP['signal'][i+20]>1.2):
        frame_time.append(i)
        i = i+3000
    i = i+5   
#%%检查阈值
for i in range(1,len(frame_time)):
    if frame_time[i]-frame_time[i-1]<5500:
        raise Exception('Frame find Error! Please recheck the threshold!\n')
frame_time = frame_time[0:(len(frame_time)-1)]#最后1帧不要了
#以上方法可以得到每一个刺激对应的时间，注意20和5这样的参数都是调出来的，对RG可能就不能用了
#%%接下来，把每一帧属于哪个序列做出来，分配frame_time于Stim_ID_Time
Frame_Stim_Check = {}
for i in range(0,len(stim_set)):
    temp_list = Stim_ID_Time[stim_set[i]]#提取出来每个刺激ID对应的序列
    temp_frame_list = []#作为符合条件的帧的容器
    for j in range(0,len(frame_time)):#将每一帧对应的时间找回去
        if frame_time[j] in temp_list:
            temp_frame_list.append(j)
    Frame_Stim_Check[stim_set[i]] = temp_frame_list
#以上是全部刺激态的id
#%%接下来把全部没刺激的frame搞出来
all_stim_frame_temp = list(Frame_Stim_Check.values())
all_stim_frame = []
for i in range(0,len(all_stim_frame_temp)):
    all_stim_frame.extend(all_stim_frame_temp[i])
#得到全部有刺激的frameid
temp_frame_list = []#重新初始化
for i in range(0,len(frame_time)):
    if (i in all_stim_frame) == False:
        temp_frame_list.append(i)
Frame_Stim_Check['Stim_Off'] = temp_frame_list
#到这里，我们可以得到每一帧对应的stimID。
#%%对Frame进行调整，每一个condition截掉前面几个，加入后面几个。
for i in range(0,len(stim_set)):
    current_condition = Frame_Stim_Check[stim_set[i]]
    all_conditions = [list(group) for group in mit.consecutive_groups(current_condition)]#把当前的condition分开，进行一些操作。
    all_conditions_adjusted = []#把调整之后的all_condition计入新的列表。
    for j in range(0,len(all_conditions)):
        all_conditions[j] = all_conditions[j][pre_stim_frame:]#截掉prestim的几帧
        for k in range(0,after_stim_drift):
            all_conditions[j].append(max(all_conditions[j])+1)#在后面加上多数的几帧。
        all_conditions_adjusted.extend(all_conditions[j])
    Frame_Stim_Check[stim_set[i]] = all_conditions_adjusted
#接下来处理Stim_Off,每个减掉开始的n个。
current_condition = Frame_Stim_Check['Stim_Off']
all_conditions = [list(group) for group in mit.consecutive_groups(current_condition)]#把当前的condition分开，进行一些操作。
all_conditions_adjusted = []#把调整之后的all_condition计入新的列表。
for j in range(0,len(all_conditions)):
    all_conditions[j] = all_conditions[j][after_stim_drift:]#截掉每个stimOff的最前面几个的几帧
    all_conditions_adjusted.extend(all_conditions[j])
Frame_Stim_Check['Stim_Off'] = all_conditions_adjusted
#%%
import pickle
fw = open((save_folder+'\\Frame_Stim_Check'),'wb')
pickle.dump(Frame_Stim_Check,fw)