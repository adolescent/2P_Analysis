
'''

This class will generate item based coding method, we give in data with calculated caiman, all results will be returned.

'''
#%% Imports 
import OS_Tools_Kit as ot
import numpy as np



#%%
class Cell(object):
    
    name = 'Single Run Cell'
    
    def __init__(self,day_folder,od_run = 6,orienrun = 2,color_run = 7,sub_folder = '_CAIMAN'):
        print('Using Cell Class for data process, make sure data have already been caimaned.')
        self.wp = ot.join(day_folder,sub_folder)
    
    def __getitem__(self,key):# return raw specific cell all run are in series.
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
    
    
#%% Test parts
if __name__ == '__main__':
    day_folder = r'E:\2P_Raws\220630_L76_2P'