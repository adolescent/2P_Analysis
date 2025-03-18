'''
Average raw graph in spontaneous folder.

'''
#%%
import OS_Tools_Kit as ot
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

all_tif = ot.Get_File_Name(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1_Repeat\L76_17B_220712\1-001')

#%% average graph
avr = np.zeros(shape = (512,512),dtype = 'f8')
for i,c_tif in tqdm(enumerate(all_tif)):
    c_graph = cv2.imread(c_tif,-1)
    c_graph = c_graph.astype('f8')
    avr += c_graph

#%% clip and normalize 
avr_c = np.clip(avr,avr.mean()-5*avr.std(),avr.mean()+5*avr.std())




