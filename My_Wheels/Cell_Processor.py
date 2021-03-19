# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:21:30 2021

@author: ZR

Define a class to include any data processing method based on cell data.
Use class so new method can be included easily.

"""

class Single_Cell_Processor(object):
    name = r'ad'
    def __init__(self,cell_dic,average_graph = None):
        self.cell_dic = cell_dic
        self.average_graph = average_graph
        self.name = self.cell_dic['Name']
        
        
class Multi_Cell_Processor(object):
    pass
    
    
