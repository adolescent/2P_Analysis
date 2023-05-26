# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:16:19 2022

@author: adolescent

This part is used to compare caiman finded dF/F data vs normal method finded data.
"""

import OS_Tools_Kit as ot
from My_FFT import FFT_Power
from Cell_Find_From_Graph import Cell_Find_And_Plot
from Standard_Cell_Generator import Standard_Cell_Generator
import Graph_Operation_Kit as gt
import cv2
import numpy as np
import matplotlib.pyplot as plt

day_folder = r'D:\Test_Data\2P\220421_L85'
all_cell_cai = ot.Load_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN\All_Series_Dic.pkl')
save_folder = r'C:\Users\adolescent\Desktop\Work_At_Home\220517_Caiman_Compare\Graphs'
#%% Generate cells
Cell_Find_And_Plot(day_folder, 'Global_Average.tif', '_Morpho')
Scg = Standard_Cell_Generator('L85', '220421', day_folder, [1,2,3,4,5,6,7,8,9],cell_subfolder = r'\_Morpho')
Scg.Generate_Cells()
#%% get number annotated caiman graphs.
import numpy as np
import My_Wheels.Graph_Operation_Kit as gt
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

all_cell_ids = list(all_cell_cai.keys())
base_graph = cv2.imread(day_folder+r'\Global_Average.tif')
font = ImageFont.truetype('arial.ttf',11)
annotated_graph = base_graph.copy().astype('f8')
for i,c_id in enumerate(all_cell_ids):
    annotated_graph[:,:,1] += all_cell_cai[c_id]['Cell_Mask']*100
annotated_graph = np.clip(annotated_graph,0,255).astype('u1')
im = Image.fromarray(annotated_graph)
for i,c_id in enumerate(all_cell_ids):
    y,x = all_cell_cai[c_id]['Cell_Loc']
    draw = ImageDraw.Draw(im)
    draw.text((x+5,y+5),str(c_id),(0,255,100),font = font,align = 'center')
final_graph = np.array(im)
gt.Show_Graph(final_graph, 'Numbers', save_folder) 


#%% Get origin method cell datas.
all_cell_dic = ot.Load_Variable(r'D:\Test_Data\2P\220421_L85\L85_220421A_All_Cells.ac')
acn_old = list(all_cell_dic.keys())
global_avr = cv2.imread(day_folder+r'\Global_Average.tif')
# Caiman 329 =morpho 88.
# Get old & new cell trains.
new_tc = all_cell_cai[329]
old_tc = all_cell_dic[acn_old[88]]
new_train = new_tc['1-001']
old_train = old_tc['Run001']['F_train']
old_dFF_train = old_train/old_train.mean()-1
new_dFF_train = new_train/new_train.mean()-1
plt.plot(old_dFF_train)
plt.plot(new_dFF_train)
# frequency analyze and filter.
from My_FFT import FFT_Power
from Filters import Signal_Filter

new_train_fft = FFT_Power(new_dFF_train)
old_train_fft = FFT_Power(old_dFF_train)
plt.plot(old_train_fft)
plt.plot(new_train_fft)

filted_new_train = Signal_Filter(new_dFF_train,filter_para = (0.005*2/1.301,0.3*2/1.301))
#filted_new_train -= filted_new_train.mean()
filted_old_train = Signal_Filter(old_dFF_train,filter_para = (0.005*2/1.301,0.3*2/1.301))
#filted_old_train -= filted_old_train.mean()
plt.plot(filted_new_train)
plt.plot(filted_old_train)

#%% Get all dF/F responses.
series_run01 = np.zeros(shape = (380,16827),dtype = 'f8')
series_run03 = np.zeros(shape = (380,6014),dtype = 'f8')
for i in range(380):
    c_F = all_cell_cai[i+1]['1-001']
    c_F_03 = all_cell_cai[i+1]['1-003']
    c_filted_F = Signal_Filter(c_F,filter_para = (0.005*2/1.301,0.3*2/1.301))
    c_filted_F_03 = Signal_Filter(c_F_03,filter_para = (0.005*2/1.301,0.3*2/1.301))
    c_dff = (c_filted_F-c_filted_F.mean())/c_filted_F.mean()
    c_dff_03 = (c_filted_F_03-c_filted_F_03.mean())/c_filted_F_03.mean()
    series_run01[i,:] = c_dff
    series_run03[i,:] = c_dff_03
# get raveled
import pandas as pd
df = pd.DataFrame(series_run01)
b = np.tile(df.columns, len(df.index))
a = np.repeat(df.index, len(df.columns))
c = df.values.ravel()
df = pd.DataFrame({'a':a, 'b':b, 'c':c})
print (df)
