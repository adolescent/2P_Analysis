# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:29:11 2022

@author: adolescent
"""


def Cell_In_Mask(all_cell_dic,cell_name_list,mask):
    '''
    Seperate cells in & out mask

    Parameters
    ----------
    all_cell_dic : (dic)
        Import all cell dic here.
    cell_name_list : (list)
        List of cells you want to plot.
    mask : (2D Array)
        Graph mask used.

    Returns
    -------
    cell_in_mask : (list)
        Cells in mask.
    cell_out_mask : (list)
        Cells out of mask.

    '''
    thresed_mask = mask>mask.mean()
    cell_in_mask = []
    cell_out_mask = []
    for i,cc in enumerate(cell_name_list):
        c_cell_info = all_cell_dic[cc]['Cell_Info']
        y,x = int(c_cell_info.centroid[0]),int(c_cell_info.centroid[1])
        if thresed_mask[y,x] == True:
            cell_in_mask.append(cc)
        else:
            cell_out_mask.append(cc)
            
    return cell_in_mask,cell_out_mask