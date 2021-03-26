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



class Single_Cell_Processor(object):
    
    def __init__(self,cell_dic,all_stim_dic,average_graph = None):
        self.cell_dic = cell_dic
        self.average_graph = average_graph
        self.name = self.cell_dic['Name']
        self.all_stim_dic = all_stim_dic 
    
    def Response_Map(self,runname,
                     Stim_ID_dics,
                     head_frame = 2,tail_frame = 3,
                     mode = 'F'
                     ):
        '''
        Generate Tang style Response map of given ids. 

        Parameters
        ----------
        runname : (str)
            Runname. Format as 'Run001' and so on.
        Stim_ID_dics : (Dic)
            Graph name and responded stim ID lists.
        head_frame :(int), optional
            How many frame before stim. The default is 2.
        tail_frame : (int), optional
            How many frame after stim. The default is 3.
        mode : ('F' or 'dF'),optional
            Which series to use. F series or dF/F series. The default is 'F'.

        Returns
        -------
        response_data : (Dic)
            Dictionary of response data of each given stim id.
        response_map : (fig file)
            plt fig data. Use savefig can save the plot.

        '''
        # Initialization
        subgraph_num = len(Stim_ID_dics)
        current_Stim_Frame_Align = self.all_stim_dic[runname]
        all_subgraph_name = list(self.all_stim_dic.keys())
        if mode == 'F':
            cell_series = self.cell_dic['F_train'][runname]
        elif mode == 'dF':
            cell_series = self.cell_dic['dF_F_train'][runname]
        else:
            raise IOError('Invalid input mode, please check.')
        # Get all condition response datas
        response_data = {}
        for i in range(subgraph_num):
            current_subgraph_name = all_subgraph_name[i]
            current_stim_id = Stim_ID_dics[current_subgraph_name]
            all_conditions = SDT.Frame_ID_Extrator_In_Conditions(current_Stim_Frame_Align, current_stim_id)
            condition_num = len(all_conditions)
            condition_length = len(all_conditions[0])
            response_data[all_subgraph_name[i]] = np.zeros(shape = (condition_num,condition_length),dtype = 'f8')
        
        # Graph Plotting
        col_num = int(np.ceil(np.sqrt(subgraph_num)))
        row_num = int(np.ceil(subgraph_num/col_num))
        fig,ax = plt.subplots(row_num,col_num,figsize = (30,28))# Initialize graphs
        fig.suptitle('Response Maps', fontsize=54)
        for i in range(subgraph_num):
            current_col = i%col_num
            current_row = i//col_num

            
            
        
        
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
        
    
        
        
        
class Multi_Cell_Processor(object):
    pass
    
    
