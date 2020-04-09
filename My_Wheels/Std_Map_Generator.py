# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:30:59 2020

@author: zhang

Generate std map for input graph sets. This process is important for spontaneous cell find.
"""

def Std_Map_Generator(all_tif_name,clip_std = 2.5,return_type = 'origin'):
    """
    Generate std map of input tifs, useful for cell finding.

    Parameters
    ----------
    all_tif_name : (list)
        List of used all tif names.
    clip_std : TYPE, optional
        DESCRIPTION. The default is 2.5.
    return_type : TYPE, optional
        DESCRIPTION. The default is 'origin'.

    Returns
    -------
    std_map : TYPE
        DESCRIPTION.

    """
    return std_map