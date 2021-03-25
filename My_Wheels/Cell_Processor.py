# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:21:30 2021

@author: ZR

Define a class to include any data processing method based on cell data.
Use class so new method can be included easily.

"""
from Stim_Dic_Tools import Frame_ID_Extractor
from My_Wheels.Standard_Parameters.Stimid_Combiner import Stim_ID_Combiner
import numpy as np
import OS_Tools_Kit as ot
import Stim_Dic_Tools as st



class Single_Cell_Processor(object):
    
    def __init__(self,cell_dic,all_stim_dic,average_graph = None):
        self.cell_dic = cell_dic
        self.average_graph = average_graph
        self.name = self.cell_dic['Name']
        self.all_stim_dic = all_stim_dic
    
    def Response_Map(self,ID_dics,ISI_frames = 2):
        subgraph_num = len(ID_dics)
        column_num = int(np.ceil(np.sqrt(subgraph_num)))
        row_num = int(np.ceil(subgraph_num/column_num))
        
        
        
        return Response_data,response_map
    
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
    
    
