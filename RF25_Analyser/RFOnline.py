# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:45:34 2020

@author: ZR

This program is used to run RF25 Online.
A support txt will be read in the same folder.

"""
import My_Wheels.OS_Tools_Kit as OS_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2
#%% First, read in config file. 
# All Read in shall be in this part to avoid bugs = =
f = open('Config.punch','r')
config_info = f.readlines()
del f
frame_folder = config_info[3][:-1]# Remove '\n'
stim_folder = config_info[6][:-1]# Remove '\n'
cap_freq = float(config_info[9])
frame_thres = float(config_info[12])
#%% Second do graph align.
save_folder = frame_folder+r'\Results'
aligned_tif_folder = save_folder+r'\Aligned_Frames'
all_tif_name = OS_Tools.Get_File_Name(frame_folder)
graph_size = np.shape(cv2.imread(all_tif_name[0],-1))
Translation_Alignment([frame_folder],align_range = 10,align_boulder = 40,big_memory_mode=True,graph_shape = graph_size)
aligned_all_tif_name = np.array(OS_Tools.Get_File_Name(aligned_tif_folder))
#%% Third, Stim Frame Align
jmp_step = int(5000//cap_freq)
_,Frame_Stim_Dic = Stim_Frame_Align(stim_folder,frame_thres = frame_thres,jmp_step = jmp_step)
#%% Forth, generate Morpho graph and find cell.
cell_Dic = Cell_Find_And_Plot(save_folder, 'Run_Average_After_Align.tif', 'Morpho_Cell')
cell_mask = (cell_Dic['Cell_Graph'][:,:,0])>0
#%% Fifth, calculate RF reaction.
RF_Data = np.zeros(shape = (5,5,2),dtype = 'f8')# use 5*5 matrix, set 0 are frames, set 1 are cells
loc_ids = np.array([1,26,51,76,101,126,151,176,201,226,251,276])
for i in range(5):# i as vector1
    for j in range(5):# j as vector2
        start_id = i*5+j
        current_keys = loc_ids+start_id
        current_loc_frame_id = []
        for k in range(len(current_keys)):
            current_loc_frame_id.extend(Frame_Stim_Dic[current_keys[k]])
        current_loc_graph_name = aligned_all_tif_name[current_loc_frame_id]
        current_graph = Graph_Tools.Average_From_File(current_loc_graph_name)
        all_cells = current_graph*cell_mask
        RF_Data[i,j,0] = current_graph.mean()
        RF_Data[i,j,1] = all_cells.mean()
# then sub average value.
frame_average = RF_Data[:,:,0].min()
cell_average = RF_Data[:,:,1].min()
prop_RF_Data = np.zeros(shape = (5,5,2),dtype = 'f8')
prop_RF_Data[:,:,0] = RF_Data[:,:,0]/frame_average -1
prop_RF_Data[:,:,1] = RF_Data[:,:,1]/cell_average -1
OS_Tools.Save_Variable(save_folder, 'Origin_RF_Data', RF_Data)
#%% Then do Spline interpolation here.
import pylab as pl
# Show graph first.
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
fval = prop_RF_Data[:,:,0]
cval = prop_RF_Data[:,:,1]
pl.figure(figsize=(10,7))
pl.subplot(1,2,1)
pl.xticks(np.arange(-5,6,1))
pl.yticks(np.arange(-5,6,1))
im1=pl.imshow(fval, extent=[1,5,5,1], cmap='inferno', interpolation='spline16', origin="upper")
from mpl_toolkits.axes_grid1 import make_axes_locatable
ax = pl.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(im1, cax=cax)
pl.title('Frame Data')
# Then cell graph.
pl.subplot(1,2,2)
pl.xticks(np.arange(-5,6,1))
pl.yticks(np.arange(-5,6,1))
im2=pl.imshow(cval, extent=[1,5,5,1], cmap='inferno', interpolation='spline16', origin="upper")
ax = pl.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(im2, cax=cax)
pl.title('Cell Data')
pl.savefig(save_folder+'\RF.png')
pl.show()

