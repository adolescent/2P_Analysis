
'''

This is the class formation of Cell information, given specific run and 
After All_Cell_Dic generation, Use this claa can process all basic operation.
This method need proper work of My_Wheel model.

'''



class Cell(object):
    
    name = 'Single Run Cell'
    
    def __init__(self,all_cell_dic,avr_graph = 'Not_Given',stim_frame_align = 'Not_Given',runname = 'Run001'):
        pass
    
    def __getitem__(self,key):# return specific cell 
        pass
    
    def __len__(self):
        pass
    
    def Generate_dFF(self,method = 'mean'): # This will generate dF/F series of given parameters. 'mean' method sub avr, 'least' method sub least 10%
        pass
    
    def Get_Spike_Train(self,thres):
        pass
    
    def Calculate_Firing_Rate(self,thres,winsize = 30):
        pass
    
    def Show_Cell(self,on_avr = False): # This show all cells out. if on_avr == True, stack cell on average graph.
        pass
    
    def Generate_Weighted_Cell(self,weight,on_avr = False): # weight need to be a pd array with name. Used for graph generation.
        pass
    
    
    