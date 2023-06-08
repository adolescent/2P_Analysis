
'''
This script contains multiple useful tools for cell class calculation.
It is actually a dependency.
'''
#%%
import numpy as np

def Single_Cell_Condition_Combiner(cell_response,Condition_Dics):

    all_map_name = list(Condition_Dics.keys())
    combined_response = {}
    for i,c_map in enumerate(all_map_name):
        c_stim_list = Condition_Dics[c_map]
        c_cond_array =  cell_response[c_stim_list[0]]
        if len(c_stim_list)>1:# meaning we need to concatenate.
            for j in range(len(c_stim_list)-1):
                c_cond_array = np.concatenate((c_cond_array,cell_response[c_stim_list[j+1]]),axis = 1)
        combined_response[c_map] = c_cond_array
    
    return combined_response

def All_Cell_Condition_Combiner(all_cell_CR_response,Condition_Dics):
    all_cell_name = list(all_cell_CR_response.keys())
    combined_allcell_response = {}
    for i,cc in enumerate(all_cell_name):
        cc_response = all_cell_CR_response[cc]
        combined_allcell_response[cc] = Single_Cell_Condition_Combiner(cc_response,Condition_Dics)
    return combined_allcell_response