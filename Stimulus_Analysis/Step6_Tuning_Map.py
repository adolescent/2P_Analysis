# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:18:07 2019

@author: ZR
绘制tuning map,用雷达图来看细胞的tuning

"""

import numpy as np
import matplotlib.pyplot as plt
import General_Functions.my_tools as pp
import multiprocessing as mp
import time

class Radar_Map():
    
    name = 'Cell Tuning Generation'
    
    def __init__(self,save_folder,spike_train,have_blank,cell_type):
        
        self.save_folder = save_folder
        self.Frame_Stim_Check = pp.read_variable(save_folder+r'\Frame_Stim_Check.pkl')
        self.have_blank = have_blank
        self.cell_type = cell_type
        self.spike_train = spike_train
        
    def condition_spikes(self):
        
        self.radar_folder = save_folder+r'\Cell_Tuning_'+self.cell_type
        pp.mkdir(self.radar_folder)
        self.stim_set = list(self.Frame_Stim_Check.keys())[:-1]#所有刺激的set，去掉了Stim-Off
        self.cell_condition_data = np.zeros(shape = (np.shape(self.spike_train)[0],len(self.stim_set)),dtype = np.float64)
        for i in range(0,np.shape(self.spike_train)[0]):#循环细胞
            for j in range(0,len(self.stim_set)):#循环全部condition
                temp_frame = self.Frame_Stim_Check[str(self.stim_set[j])]#当前condition的全部帧id
                self.cell_condition_data[i,j] = self.spike_train[i,temp_frame[:]].mean()   
    
    def pool_set(self):
        self.pool = mp.Pool(10)
    
    def Axis_Define(self):
        
        if have_blank ==True:
            self.cell_condition_data = self.cell_condition_data[:,1:]#去掉conditionid的影响。
            self.feature = self.stim_set[1:]
        else:
            self.feature = self.stim_set
        
    def Graph_plot(self,i):
        values = self.cell_condition_data[i,:]#每个维度的值
        self.feature = self.feature#每个维度的标签,如有特殊需要可以在这里直接定义
        N = len(values)# 设置雷达图的角度，用于平分切开一个圆面
        angles=np.linspace(0, 2*np.pi, N, endpoint=False)
        # 为了使雷达图一圈封闭起来，需要下面的步骤
        values=np.concatenate((values,[values[0]]))
        angles=np.concatenate((angles,[angles[0]]))
        # 绘图
        fig=plt.figure(figsize = (12,12))
        # 这里一定要设置为极坐标格式
        ax = fig.add_subplot(111, polar=True)
        # 绘制折线图
        ax.plot(angles, values, '', linewidth=2)
        # 填充颜色
        ax.fill(angles, values, alpha=0.25)
        # 添加每个特征的标签
        ax.set_thetagrids(angles * 180/np.pi, self.feature,fontsize = 20)
        plt.yticks(np.arange(-0.2,1,step=0.05),fontsize=8)
        # 设置雷达图的范围
        ax.set_ylim(self.cell_condition_data.min(),self.cell_condition_data.max())
        # 添加标题
        plt.title('Radar_Map_Cell'+str(i))
        # 添加网格线
        ax.grid(True)
        # 显示图形
        plt.savefig(self.radar_folder+'\Radar_Map_Cell'+str(i)+'.png')
        #plt.show()
        plt.close('all')
        
        
    def __getstate__(self):#为避免pickle出错必须要写
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        
    def main(self):
        self.condition_spikes()
        self.Axis_Define()
        self.pool_set()
        self.pool.map(self.Graph_plot,range(np.shape(self.cell_condition_data)[0]))
        self.pool.close()
        self.pool.join()
        
        
if __name__ == '__main__':
    
    start_time = time.time()
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-004\results'
    have_blank = True
    cell_type = 'Morphology'
    spike_train = pp.read_variable(save_folder+r'\spike_train_'+cell_type+'.pkl')
    rm = Radar_Map(save_folder,spike_train,have_blank,cell_type)
    rm.main()
    pp.save_variable(rm.cell_condition_data,save_folder+r'\Cell_Tunings_'+cell_type+r'.pkl')    
    finish_time = time.time()
    print('Plot Done, time cost :'+str(finish_time-start_time)+'s')