# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:49:10 2019

@author: ZR
"""

import cv2
import numpy as np
import functions_OD as pp
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt


def read_variable(name):#读取变量用的题头，希望这个可以在后续删掉
    with open(name, 'rb') as file:
        variable = pickle.load(file)
    file.close()
    return variable
#%%
class Spike_Train():
    name =r'dF/F Train Calculation'
    
    def __init__(self,cell_group,aligned_frame_name,graph_after_align,save_folder,core_Num):
        self.cell_group = cell_group
        self.aligned_frame_name = aligned_frame_name
        self.graph_after_align = graph_after_align
        self.core_Num = core_Num
        self.save_folder = save_folder
        
    def base_define(self):#得到基准F值，用来计算dF/F
        self.base_F = np.zeros(shape = len(self.cell_group))
        self.spike_train = np.zeros(shape = (len(self.cell_group),len(self.aligned_frame_name)))
        for i in range(0,len(self.cell_group)):
            self.base_F[i] = pp.sum_a_frame(self.graph_after_align,self.cell_group[i])
            
    def pool_set(self):#将进程池设为成员变量，这样就可以在dill的时候删除了
        self.pool = mp.Pool(self.core_Num)
        
    def dF_calculation(self,frame_i):
#        for frame_i in range(0,len(self.aligned_frame_name)):# 减少读取次数，每次将一张图的细胞数目读取完
        current_frame = cv2.imread(self.aligned_frame_name[frame_i],-1)
        target_F = np.zeros(shape = len(self.cell_group))
        for j in range(0,len(self.cell_group)):
            target_F[j] = pp.sum_a_frame(current_frame,self.cell_group[j])
        dF_per_frame = (target_F-self.base_F)/self.base_F
        return dF_per_frame
        #self.spike_train[:,frame_i] =dF_per_frame 
        #print(dF_per_frame[4])
        
    def plot_initialize(self):#进行绘图的初始化准备
        self.spike_train_folder = self.save_folder+'\Spike_Trains'
        pp.mkdir(self.spike_train_folder)
        self.cell_Num,self.frame_Num = np.shape(self.spike_train)
    def plot_spike_train(self,i):#对每个细胞画图，是OI操作
        plt.figure(figsize = (20,3))
        plt.ylim(-0.5,1.5)
        plt.xlim(0,self.frame_Num*1.06)
        plt.title('Spike_Train of Cell '+str(i))
        plt.plot(self.spike_train[i,:])
        #接下来标注均值和1倍、2倍标准差的线。
        mean = np.mean(self.spike_train[i,:])
        std = np.std(self.spike_train[i,:])
        plt.axhline(y = mean,color = '#d62728')#红色
        plt.annotate('Average', xy = (self.frame_Num,mean),xytext=(self.frame_Num*1.03,mean+0.5),arrowprops=dict(facecolor='black',width = 1,shrink = 0.05,headwidth = 5))#标注均值
        plt.axhline(y = mean+std*2,color = '#00ff00')#绿色
        plt.annotate('Means+2std', xy = (self.frame_Num,std*2),xytext=(self.frame_Num*1.03,std*2+0.5),arrowprops=dict(facecolor='black',width = 1,shrink = 0.05,headwidth = 5))#标注阈值
        plt.axhline(y = mean+std*3,color = '#66fff2')#水蓝色
        plt.annotate('Means+3std', xy = (self.frame_Num,std*3),xytext=(self.frame_Num*1.03,std*3+0.5),arrowprops=dict(facecolor='black',width = 1,shrink = 0.05,headwidth = 5))#标注阈值
        #%接下来计算超过2std和3std的帧数比例
        over1std = str(round(np.sum(self.spike_train[i,:]>mean+std*1)/self.frame_Num*100,5))+'%'
        over2std = str(round(np.sum(self.spike_train[i,:]>mean+std*2)/self.frame_Num*100,5))+'%'#百分数形式保留五位小数并转换为str
        plt.annotate((over1std+' Over1std,'+over2std+' Over2std'),xy = (self.frame_Num*0.75,0), xytext=(self.frame_Num*0.75,1.6))
        plt.savefig(self.spike_train_folder+'\Cell'+str(i)+'.png')
        #plt.show()
        plt.close('all')
        
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
        
    def main(self):#这里定义主函数
        self.base_define()
        self.pool_set()
        print('Calculating dF/F ...\n')
        dF_lists = st.pool.map(st.dF_calculation,range(0,len(st.aligned_frame_name)))
        for i in range(0,np.shape(self.spike_train)[1]):
            self.spike_train[:,i] = dF_lists[i]
        self.pool.close()
        self.pool.join()
        print('Calculation Done, Plotting...\n')
        self.plot_initialize()
        self.pool_set()
        self.pool.map(self.plot_spike_train,range(0,self.cell_Num))
        self.pool.close()
        self.pool.join()
        
        
    #%%    
if __name__ == '__main__':
    start_time = time.time()
    print('Spike_Train Calculating...\n')
    cell_group = read_variable('cell_group.pkl')
    aligned_frame_name = read_variable('aligned_frame_name.pkl')
    graph_after_align = read_variable('graph_after_align.pkl')
    save_folder = read_variable('save_folder.pkl')
    st = Spike_Train(cell_group,aligned_frame_name,graph_after_align,save_folder,3)
    st.main()
    spike_train = st.spike_train
    st.save_variable(spike_train,'spike_train.pkl')  
    print('Calculation Done!\n')
    finish_time = time.time()
    print('Task Time Cost:'+str(finish_time-start_time)+'s')