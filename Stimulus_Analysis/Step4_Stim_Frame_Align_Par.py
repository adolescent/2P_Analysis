# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:54:58 2019

@author: ZR
"""
import os
import neo
import numpy as np
import pickle
import time
import multiprocessing as mp
import more_itertools as mit
import General_Functions.my_tools as pp


#%%

class Stim_Frame_Align():
    
    name =r'Frame_Stim_Align'
    
    def __init__(self,stim_folder,pre_stim_frame,after_stim_drift):
        self.stim_folder = stim_folder
        self.pre_stim_frame = pre_stim_frame
        self.after_stim_drift = after_stim_drift
        self.Frame_Stim_Check = {}

    def stim_file_name(self): 
        #首先遍历得到文件名
        for root, dirs, files in os.walk(self.stim_folder):
            for file in files:
                if root == self.stim_folder:#只遍历根目录，不操作子目录的文件
                    if os.path.splitext(file)[1] == '.txt':
                        txt_name = (os.path.join(root, file))
                    if os.path.splitext(file)[1] == '.smr':
                        smr_name = (os.path.join(root, file))
        #再读文件名得到文件内容。
        reader = neo.io.Spike2IO(filename=smr_name)
        self.smr_data = reader.read(lazy=False)[0]
        f = open(txt_name,'r')
        self.stim_id = f.read().split()
        f.close()
        
    def spike2_series_extract(self):
        #读取两个spike2的数据序列，并保存为字典。
        self.Stimuli = {}
        self.twoP = {}#将两个序列存入字典，包括了时间、信号序列、采样频率。
        for i, asig in enumerate(self.smr_data.segments[0].analogsignals):
                # 提取时间，并化为秒作单位。
            times = asig.times.rescale('s').magnitude
            # Determine channel name, without leading b' 
            ch = str(asig.annotations['channel_names']).split(sep="'")[1]
            # Extract sampling frequency
            fs = float(asig.sampling_rate)
            # Assign sampling times, sampling frequency and data to correct dictionary
            if ch == 'Stimuli':
                self.Stimuli['times'] = times #时间序列
                self.Stimuli['signal'] = np.array(asig) #信号序列
                self.Stimuli['fs'] = fs #采样频率
            elif ch == '2P':
                self.twoP['times'] = times #时间序列
                self.twoP['signal'] = np.array(asig) #信号序列
                self.twoP['fs'] = fs  #采样频率
    def Stim_Time_Align(self):#将txt文件的id对应的stim time进行对齐
        all_stim_time = [x for x in range(len(self.Stimuli['signal'])) if self.Stimuli['signal'][x] > 2]#将信号大于2V的认为是刺激ON
#        import more_itertools as mit
        stim_lists = [list(group) for group in mit.consecutive_groups(all_stim_time)]
        #定义一个异常，以免出现计算错误
        if len(stim_lists)!=len(self.stim_id):
            raise Exception('Stim sequence not match, please recheck stim ids.\n')
        #将stim_id与stim_lists相对应，并将刺激id对应的时间归到一个字典文件里。
        self.Stim_ID_Time = {}#定义空字典，准备加入每个id和它对应的时间序列。
        self.stim_set = list(set(self.stim_id))
        self.stim_set.sort()
        #遍历每一个刺激id，得到每个刺激id对应的时间。
        for i in range(0,len(self.stim_set)):
            temp_index = [j for j in range(len(self.stim_id)) if self.stim_id[j] == self.stim_set[i]]#这里得到的是所有stim_id为i的序列。
            temp_stim_time = []#定义空列表，作为每一个刺激id的时间暂存器
            for k in range(0,len(temp_index)):
                temp_stim_time.extend(stim_lists[temp_index[k]])
            self.Stim_ID_Time[self.stim_set[i]] = temp_stim_time#
        #到此为止，我们得到了每个刺激id播放时对应的时间。
    def Frame_Time_Align(self):#找到每一帧的开始时间，并把全部帧的时间记录下来
        self.frame_time = []#定义空列表，记录每一帧播放时在spike2上记录的时间。
        i = 20
        while i<(len(self.Stimuli['times'])-20):
            if (self.twoP['signal'][i-20]<0 and self.twoP['signal'][i+20]>1.2):
                self.frame_time.append(i)
                i = i+3000
            i = i+5   
        #检查阈值
        for i in range(1,len(self.frame_time)):
            if self.frame_time[i]-self.frame_time[i-1]<5500:
                raise Exception('Frame find Error! Please recheck the threshold!\n')
        self.frame_time = self.frame_time[0:(len(self.frame_time)-1)]#最后1帧不要了
        #以上方法可以得到每一个刺激对应的时间，注意20和5这样的参数都是调出来的，对RG可能需要修改
        
    def pool_set(self,core_Num):#将进程池设为成员变量，这样就可以在pickle的时候删除了
        self.pool = mp.Pool(core_Num)
        
    def Stim_Frame_find(self,i):#对齐，即分配每帧对应的刺激id
            #for i in range(0,len(self.stim_set)):
        temp_list = self.Stim_ID_Time[self.stim_set[i]]#提取出来每个刺激ID对应的序列
        temp_frame_list = []#作为符合条件的帧的容器
        for j in range(0,len(self.frame_time)):#将每一帧对应的时间找回去
            if self.frame_time[j] in temp_list:
                temp_frame_list.append(j)
        return temp_frame_list
       # Frame_Stim_Check[self.stim_set[i]] = temp_frame_lists
       
    def run_Stim_Frame_find(self):
        self.pool_set(9)
        frame_lists = self.pool.map(self.Stim_Frame_find,range(0,len(self.stim_set)))#跑成一个列表，从头到尾是每个stim对应的id
        self.pool.close()
        self.pool.join()
        for i in range(0,len(frame_lists)):
            self.Frame_Stim_Check[self.stim_set[i]] = frame_lists[i]
   
    def Stim_Off_Get(self):#在以上ID 的情况下找到stim_Off 对应的帧，并写入文件。
        all_frame_lists = list(range(len(self.frame_time)))#这个是全部的帧,从中去掉属于至少一个condition的就是isi的
        stim_frames_temp = list(self.Frame_Stim_Check.values())
        stim_frame = []
        for i in range(0,len(stim_frames_temp)):
            stim_frame.extend(stim_frames_temp[i])
        no_stim_frame = [x for x in all_frame_lists if x not in stim_frame] #这里定义没有刺激的帧
        self.Frame_Stim_Check['Stim_Off'] = no_stim_frame
        #%%到原来98行
    def Frame_adjust(self):
        #%%对Frame进行调整，每一个condition截掉前面几个，加入后面几个。
        for i in range(0,len(self.stim_set)):
            current_condition = self.Frame_Stim_Check[self.stim_set[i]]
            all_conditions = [list(group) for group in mit.consecutive_groups(current_condition)]#把当前的condition分开，进行一些操作。
            all_conditions_adjusted = []#把调整之后的all_condition计入新的列表。
            for j in range(0,len(all_conditions)):
                all_conditions[j] = all_conditions[j][self.pre_stim_frame:]#截掉prestim的几帧
                for k in range(0,after_stim_drift):
                    all_conditions[j].append(max(all_conditions[j])+1)#在后面加上多数的几帧。
                all_conditions_adjusted.extend(all_conditions[j])
            self.Frame_Stim_Check[self.stim_set[i]] = all_conditions_adjusted
        #接下来处理Stim_Off,每个减掉开始的n个。
        current_condition = self.Frame_Stim_Check['Stim_Off']
        all_conditions = [list(group) for group in mit.consecutive_groups(current_condition)]#把当前的condition分开，进行一些操作。
        all_conditions_adjusted = []#把调整之后的all_condition计入新的列表。
        for j in range(0,len(all_conditions)):
            all_conditions[j] = all_conditions[j][after_stim_drift:]#截掉每个stimOff的最前面几个的几帧
            all_conditions_adjusted.extend(all_conditions[j])
        self.Frame_Stim_Check['Stim_Off'] = all_conditions_adjusted
        
    def __getstate__(self):#为避免pickle出错必须要写
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def save_variable(self,variable,name):
        fw = open(name,'wb')
        pickle.dump(variable,fw)#保存细胞连通性质的变量。 
        fw.close()
            
        
        
#%%        
if __name__ == '__main__':
    
    #Attention! Only 1 txt file in stim folder is acceptable.
    start_time = time.time()
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-004\results'
    #save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\190412_L74_stimuli\Run02_2P_G8\test'
    stim_folder = r'E:\ZR\Data_Temp\190412_L74_LM\190412_L74_stimuli\Run04_2P_RGLum4'
    pre_stim_frame = 0#这里指的是方波开始刺激没放，需要删除的帧数。
    after_stim_drift = 0#这里指的是锯齿波消失之后，再计算几帧属于其中。
    sf= Stim_Frame_Align(stim_folder,pre_stim_frame,after_stim_drift)
    sf.stim_file_name()
    sf.spike2_series_extract()
    smr_data_2p = sf.twoP
    smr_data_Stim = sf.Stimuli
    stim_id = sf.stim_id
    sf.Stim_Time_Align()
    sf.Frame_Time_Align()
    Stim_ID_Time = sf.Stim_ID_Time
    Frame_time = sf.frame_time
    print('Stim Frame Aligning...\n')
    sf.run_Stim_Frame_find()
    sf.Stim_Off_Get()
    sf.Frame_adjust()
    pp.save_variable(sf.Frame_Stim_Check,save_folder+r'\Frame_Stim_Check.pkl')
    
    finish_time = time.time()
    print('Task Time Cost:'+str(finish_time-start_time)+'s')
    