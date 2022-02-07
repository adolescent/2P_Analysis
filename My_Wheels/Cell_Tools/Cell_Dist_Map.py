# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:39:02 2022

@author: ZR
"""

import OS_Tools_Kit as ot
import pandas as pd
import numpy as np
from tqdm import tqdm

#%%
def Cell_Dist_Map(day_folder):
    
    all_cell_dic = ot.Load_Variable(ot.Get_File_Name(day_folder,'.ac')[0])
    acn = list(all_cell_dic.keys())
    ac_center = pd.DataFrame(index = acn,columns = ['y','x'])
    for i,cc in enumerate(acn):
        tc = all_cell_dic[cc]
        ac_center.loc[cc,'y'],ac_center.loc[cc,'x'] = tc['Cell_Info'].centroid
    del all_cell_dic
    cell_dist_frame = pd.DataFrame(index = acn,columns = acn)
    for i,cell_A in tqdm(enumerate(acn)):
        for j,cell_B in enumerate(acn):
            A_cord = ac_center.loc[cell_A]
            B_cord = ac_center.loc[cell_B]
            vec = A_cord-B_cord
            c_dist = np.sqrt((vec*vec).sum())
            cell_dist_frame.loc[cell_A,cell_B] = c_dist
    cell_dist_frame = cell_dist_frame.fillna(0)
    
    return cell_dist_frame




