# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:41:57 2022

@author: adolescent

This script will change 220421 file into new caimanned result.
"""


import cv2
from Caiman_API.One_Key_Caiman import One_Key_Caiman
# First, align graphs with Run03.
day_folder = r'D:\Test_Data\2P\220421_L85'
base_graph = cv2.imread(r'D:\Test_Data\2P\220421_L85\_CAIMAN_old\Align_Template.tif',0)
Okc = One_Key_Caiman(day_folder, [1,2,3,7,8,9])
all_runname = Okc.all_stack_names
for i,c_run in enumerate(all_runname):
    Okc.Motion_Corr_Single(c_run,tamplate = base_graph)
    
#%% Get cell graph from each file.
import OS_Tools_Kit as ot
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import numpy as np
import caiman as cm
from tqdm import tqdm

all_mmap_name = ot.Get_File_Name(r'D:\Test_Data\2P\220421_L85\_CAIMAN','.mmap')
cell_id_dic = ot.Load_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN_old','comp_id_dic.pkl')
cnm2 = load_CNMF(r'D:\Test_Data\2P\220421_L85\_CAIMAN_old\analysis_results.hdf5')
cnm2.estimates.plot_contours_nb()
# Cell mask here.
cell_series_dic = {}
for i in range(len(cell_id_dic)):
    cell_series_dic[i+1] = {}
    cc = cell_id_dic[i+1]
    c_mask = np.reshape(cnm2.estimates.A[:,cc].toarray(), (512,512), order='F')
    c_loc = cnm2.estimates.coordinates[cc]['CoM']
    cell_series_dic[i+1]['Cell_Loc'] = c_loc
    cell_series_dic[i+1]['Cell_Mask'] = c_mask

#%% Calculate and save cell responses.
for k,c_run in enumerate(all_mmap_name):
    c_subname = c_run.split('\\')[-1][:5]
    Yr, dims, T = cm.load_memmap(c_run)
    c_images = np.reshape(Yr.T, [T] + list(dims), order='F')
    c_frame_num = c_images.shape[0]
    group_step = 6200
    group_num = np.ceil(c_frame_num/group_step).astype('int')
    all_cell_data = np.zeros(shape = (len(cell_id_dic),c_frame_num),dtype = 'f8')
    for i in tqdm(range(group_num)):
        if i != group_num-1:# not the last group
            c_frame_group = np.array(c_images[i*group_step:(i+1)*group_step,:,:])
            for j in tqdm(range(len(cell_id_dic))):
                c_mask = cell_series_dic[j+1]['Cell_Mask']
                cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                all_cell_data[j,i*group_step:(i+1)*group_step] = cc_resp
            del c_frame_group
        else:# the last group
            c_frame_group = np.array(c_images[i*group_step:,:,:])
            for j in tqdm(range(len(cell_id_dic))):
                c_mask = cell_series_dic[j+1]['Cell_Mask']
                cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                all_cell_data[j,i*group_step:] = cc_resp 
            del c_frame_group
            
    # save in dics.
    for i in range(len(cell_id_dic)):
        
        cc_series_all = all_cell_data[i,:]
        cell_series_dic[i+1][c_subname] = cc_series_all
        
ot.Save_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN', 'All_Series_Dic', cell_series_dic)
        
        
    
    
    
    
    


