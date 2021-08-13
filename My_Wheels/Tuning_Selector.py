# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:31:16 2021

@author: ZR
"""

import OS_Tools_Kit as ot


def Get_Tuning_Cells(day_folder,tuning_lists):
    
    tuning_file_name = ot.Get_File_Name(day_folder,'.tuning')[0]
    tuning_file = ot.Load_Variable(tuning_file_name)
    all_cell_name = list(tuning_file.keys())
    tuned_cells = []
    for i,c_name in enumerate(all_cell_name):
        for j,c_tuning in enumerate(tuning_lists):
            if c_tuning in tuning_file[c_name]['_Significant_Tunings']:
                tuned_cells.append(c_name)
                break
    return tuned_cells




if __name__ == '__main__':
    
    day_folder = r'K:\Test_Data\2P\210629_L76_2P'
    


