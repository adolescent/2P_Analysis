# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:18:31 2020

@author: ZR
"""

def Sub_Dic_Generator(mode):
    '''
    Generate subtraction dics.

    Parameters
    ----------
    mode : ('G8','G8+90','RGLum4','OD_OI','OD_2P')
        Mode of data. This is important for 

    Returns
    -------
    sub_dics : TYPE
        DESCRIPTION.

    '''
    sub_dics = {}
    if mode == 'OD_OI':
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['OD'] = [[1,2,3,4],[5,6,7,8]]
        sub_dics['L-0'] = [[1,2,3,4],[0]]
        sub_dics['R-0'] = [[5,6,7,8],[0]]
        sub_dics['A-O'] = [[4,8],[2,6]]
        sub_dics['H-V'] = [[3,7],[1,5]]
        sub_dics['HV-AO'] = [[1,3,5,7],[2,4,6,8]]
        
    elif mode == 'OD_2P':
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['OD'] = [[1,3,5,7],[2,4,6,8]]
        sub_dics['L-0'] = [[1,3,5,7],[0]]
        sub_dics['R-0'] = [[2,4,6,8],[0]]
        sub_dics['H-V'] = [[1,2],[5,6]]
        sub_dics['A-O'] = [[3,4],[7,8]]
        sub_dics['HV-AO'] = [[1,2,5,6],[3,4,7,8]]
        
    elif mode == 'G8_Norm':# Normal G8, ID1 is right moving bars.
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['H-V'] = [[3,7],[1,5]]
        sub_dics['A-O'] = [[4,8],[2,6]]
        sub_dics['HV-AO'] = [[1,3,5,7],[2,4,6,8]]
        sub_dics['Orien0-0'] = [[3,7],[0]]
        sub_dics['Orien45-0'] = [[4,8],[0]]
        sub_dics['Orien90-0'] = [[1,5],[0]]
        sub_dics['Orien135-0'] = [[2,6],[0]]
        sub_dics['DirU-D'] = [[2,3,4],[6,7,8]]
        sub_dics['DirL-R'] = [[4,5,6],[8,1,2]]
        
    elif mode == 'G8+90': # In early 2p stims, use up moving bars as ID1.
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['H-V'] = [[1,5],[3,7]]
        sub_dics['A-O'] = [[2,6],[4,8]]
        sub_dics['HV-AO'] = [[1,3,5,8],[2,4,6,8]]
        sub_dics['Orien0-0'] = [[1,5],[0]]
        sub_dics['Orien45-0'] = [[2,6],[0]]
        sub_dics['Orien90-0'] = [[3,7],[0]]
        sub_dics['Orien135-0'] = [[4,8],[0]]
        sub_dics['DirU-D'] = [[2,1,8],[4,5,6]]
        sub_dics['DirL-R'] = [[2,3,4],[6,7,8]]
        
        
    elif mode == 'RGLum4':
        sub_dics['All-0'] = [[1,2,3,4],[0]]
        sub_dics['RG-Lum'] = [[1,2],[3,4]]
        sub_dics['A-O'] = [[1,3],[2,4]]
        sub_dics['RG-0'] = [[1,2],[0]]
        sub_dics['Lum-0'] = [[3,4],[0]]
        
        
    return sub_dics