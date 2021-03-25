# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:21:30 2021

@author: ZR

Define a class to include any data processing method based on cell data.
Use class so new method can be included easily.

"""
from Stim_Dic_Tools import Frame_ID_Extractor



class Single_Cell_Processor(object):
    
    def __init__(self,cell_dic,all_stim_dic,average_graph = None):
        self.cell_dic = cell_dic
        self.average_graph = average_graph
        self.name = self.cell_dic['Name']
        self.all_stim_dic = all_stim_dic
        
    
        
        
        
class Multi_Cell_Processor(object):
    pass
    
    
