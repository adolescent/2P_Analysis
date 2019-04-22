# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:37:05 2019

@author: ZR
"""
import numpy as np
import matplotlib.pyplot as plt
#%%首先得到每个cell对各个condition的dF/F平均。
#注意，如果刺激没有0id的话，这里需要做一些修改。
have_blank = True#condition里是否有condition0，如果有的话则为真，否则为假。
radar_folder = save_folder+r'\Radar_Map'
pp.mkdir(radar_folder)
cell_condition_data = np.zeros(shape = (np.shape(spike_train)[0],len(stim_set)),dtype = np.float64)#得到每个细胞的condition tuning 数据
for i in range(0,np.shape(spike_train)[0]):#循环细胞
    for j in range(0,len(stim_set)):#循环全部condition
        temp_frame = Frame_Stim_Check[str(stim_set[j])]#当前condition的全部帧id
        cell_condition_data[i,j] = spike_train[i,temp_frame[:]].mean()
    
#%% 接下来进行雷达图的绘制。
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
# 使用ggplot的绘图风格
plt.style.use('ggplot')
# 构造数据
if have_blank ==True:
    cell_condition_data = cell_condition_data[:,1:]#去掉conditionid的影响。
    feature = list(Frame_Stim_Check.keys())[1:(len(Frame_Stim_Check)-1)]
else:
    feature = list(Frame_Stim_Check.keys())[0:(len(Frame_Stim_Check)-1)]
for i in range(0,np.shape(spike_train)[0]):#循环全部细胞
    values = cell_condition_data[i,:]#每个维度的值
    feature = feature#每个维度的标签,如有特殊需要可以在这里直接定义
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
    ax.set_thetagrids(angles * 180/np.pi, feature,fontsize = 20)
    plt.yticks(np.arange(-0.2,1,step=0.05),fontsize=8)
    # 设置雷达图的范围
    ax.set_ylim(cell_condition_data.min(),cell_condition_data.max())
    # 添加标题
    plt.title('Radar_Map_Cell'+str(i))
    # 添加网格线
    ax.grid(True)
    # 显示图形
    plt.savefig(radar_folder+'\Radar_Map_Cell'+str(i)+'.png')
    plt.show()
    plt.close('all')