# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:21:30 2021

@author: ZR

Define a class to include any data processing method based on cell data.
Use class so new method can be included easily.

"""
import  Stim_Dic_Tools as SDT
from My_Wheels.Standard_Parameters.Stimid_Combiner import Stim_ID_Combiner
import numpy as np
import OS_Tools_Kit as ot
import Stim_Dic_Tools as st
import matplotlib.pyplot as plt
import numpy as np
import My_Wheels.Filters as Filters


class Cell_Processor(object):
    
    def __init__(self,cell_data_path,
                 all_stim_dic_path,
                 runname,
                 average_graph = None,
                 series_mode = 'F',
                 filter_para = (0.02,False)
                 ):
        # Read in variables
        self.all_cell_dic = ot.Load_Variable(cell_data_path)
        self.average_graph = average_graph
        self.all_stim_dic = ot.Load_Variable(all_stim_dic_path)
        self.stim_frame_align = self.all_stim_dic[runname]
        # Then get each cell spike train.
        cell_num = len(self.all_cell_dic)
        self.all_cells_train = {}
        self.all_cell_names = list(self.all_cell_dic.keys())
        for i in range(cell_num):
            current_name = self.all_cell_names[i]
            if series_mode == 'F':
                cell_series = self.all_cell_dic[current_name]['F_train'][runname]
            elif series_mode == 'dF':
                cell_series = self.all_cell_dic[current_name]['dF_F_train'][runname]
            else:
                raise IOError('Invalid input mode, please check.')
            cell_series = Filters.Signal_Filter(cell_series,filter_para = filter_para)# Then filter cell train
            self.all_cells_train[current_name] = cell_series

    def Single_Cell_Response_Map(self,Condition_dics,
                     cell_name,
                     head_frame = 3,tail_frame = 3,
                     std_annotate = True
                     ):

        # Initialization
        subgraph_num = len(Condition_dics)
        all_subgraph_name = list(Condition_dics.keys())
        cell_train = self.all_cells_train[cell_name]
        response_data = {}
        # For every map.
        for i in range(subgraph_num):
            current_subgraph_name = all_subgraph_name[i]
            current_stim_id = Condition_dics[current_subgraph_name]
            all_conditions = SDT.Frame_ID_Extrator_In_Conditions(self.stim_frame_align, 
                                                                 current_stim_id,
                                                                 head_extend= head_frame,
                                                                 tail_extend=tail_frame)
            condition_num = len(all_conditions)
            condition_length = len(all_conditions[0])
            response_data[all_subgraph_name[i]] = np.zeros(shape = (condition_num,condition_length),dtype = 'f8')
            # Fill in response data matrix.
            for j in range(condition_num):
                current_frame_id = all_conditions[j]
                response_data[all_subgraph_name[i]][j,:] = cell_train[current_frame_id]
            
            
        # Graph Plotting
        col_num = int(np.ceil(np.sqrt(subgraph_num)))
        row_num = int(np.ceil(subgraph_num/col_num))
        fig,ax = plt.subplots(row_num,col_num,figsize = (30,28))# Initialize graphs
        fig.suptitle('Response Maps', fontsize=54)
        for i in range(subgraph_num):
            current_col = i%col_num
            current_row = i//col_num
            
            
            
        
        response_map = None
        return response_data,response_map
    
    def Wavelet_Specs(self):
        # This is an example. 
        import pywt
        import numpy as np
        import matplotlib.pyplot as plt
        x = np.arange(512)
        y = np.sin(2*np.pi*x/32)
        coef, freqs=pywt.cwt(y,np.arange(1,512),'cgau7')
        plt.matshow(coef) # doctest: +SKIP
        plt.show()
    def Cross_Wavelet(self):
        # Cross power specturn calculator
        pass
        
    
        
        
        

    
