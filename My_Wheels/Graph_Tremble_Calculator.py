# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:34:33 2020

@author: ZR
Calculate graph tremble. Return tramle plot.
"""
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Cutter as Cutter

def Tremble_Calculator_From_File(
        data_folder,
        cut_shape = (10,5),
        move_method = 'average',
        base = [],
        center_method = 'weight'
        ):
    '''
    Calculate align tremble from graph. This program is used to evaluate align quality.
    
    Parameters
    ----------
    data_folder : (str)
        Data folder of graphs.
    cut_shape : (turple), optional
        Shape of fracture cut. Proper cut will . The default is (10,5).
    move_method : ('average'or'former'or'input'), optional
        Method of bais calculation. 
        'average' bais use all average; 'former' bais use fomer frame; 'input' bais need to be given.
        The default is 'average'.
    base : (2D_NdArray), optional
        If move_method == 'input', base should be given here. The default is [].
    center_method : ('weight' or 'binary'), optional
        Method of center find. Whether we use weighted intense.The default is 'weight'.

    Returns
    -------
    tremble_plots : (List)
        List of all fracture graph tremble list.
    tremble_information : (Dic)
        Dictionary of tramble informations.

    '''
    
    return tremble_plots,tremble_information