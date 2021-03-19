# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:52:17 2021

@author: ZR
This part is used to generate standarized data type of cells. This is very useful in data processing
"""
import My_Wheels.OS_Tools_Kit as ot
import My_Wheels.List_Operation_Kit as lt
from My_Wheels.Spike_Train_Generator import Single_Cell_Spike_Train

def Standard_Cell_Processor(
        animal_name,
        date,
        day_folder,
        cell_file_path,
        average_graph_path,
        run_id_lists,
        location = 'A',# For runs have 
        Stim_Frame_Align_subfolder = r'\Results\Stim_Frame_Align.pkl',# if not read, regard as spon runs.
        align_subfolder = r'\Results\Aligned_Frames',  
        ):
    # Folder and name initialization
    print('Just make sure average and cell find is already done.')
    cell_dic = ot.Load_Variable(cell_file_path)
    cell_info = cell_dic['All_Cell_Information']
    cell_name_prefix = animal_name+'_'+str(date)+location+'_'
    all_cell_num = len(cell_info)
    all_run_subfolders = lt.List_Annex([day_folder], lt.Run_Name_Producer_2P(run_id_lists))
    save_folder = day_folder+r'\_Cell_Data'
    ot.mkdir(save_folder)
    # Then calculate each cell dic file.
    for i in range(all_cell_num):
        current_cell_name = cell_name_prefix+ot.Bit_Filler(i,4)
        current_cell_dic = {}
        current_cell_dic['Name'] = current_cell_name
        current_cell_dic['Cell_Info'] = cell_info[i]
        # Cycle all runs for F and dF trains.
        current_cell_dic['dF_F_train'] = {}
        current_cell_dic['F_train'] = {}
        for j in range(len(all_run_subfolders)):
            current_runid = 'Run'+(all_run_subfolders[j][-3:])# Use origin run id to avoid bugs.
            current_all_tif_name = ot.Get_File_Name(all_run_subfolders[j]+align_subfolder,'.tif')
            current_Stim_Frame_Align = ot.Load_Variable(all_run_subfolders[j]+Stim_Frame_Align_subfolder)
            if current_Stim_Frame_Align == False : # meaning this run is spon.
                current_F,current_dF_F = Single_Cell_Spike_Train(current_all_tif_name, cell_info[i],Base_F_type='most_unactive',stim_train = None)
            else:
                current_run_stim_train = current_Stim_Frame_Align['Original_Stim_Train']
                if 0 in current_run_stim_train:# having 0
                    current_F,current_dF_F = Single_Cell_Spike_Train(current_all_tif_name, cell_info[i],Base_F_type='nearest_0',stim_train = current_run_stim_train)
                else:
                    current_F,current_dF_F = Single_Cell_Spike_Train(current_all_tif_name, cell_info[i],Base_F_type='before_ISI',stim_train = current_run_stim_train)
            current_cell_dic['dF_F_train'][current_runid] = current_dF_F
            current_cell_dic['F_train'][current_runid] = current_F
        # Then save current cell.
        ot.Save_Variable(save_folder, current_cell_name, current_cell_dic,'.sc')
            