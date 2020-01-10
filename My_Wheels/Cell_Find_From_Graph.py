# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:32:02 2020

@author: ZR

INPUT: Base Graph, use this graph to find cell, multi parameter adjustable.

OUTPUT: Cell_Group, an skimage data, variable, with 
        Cell_Graph, Annotate cells, with and without label number
"""
import My_Wheels.Calculation_Functions as Calculator
import scipy.ndimage
import numpy as np
import My_Wheels.Graph_Operation_Kit as Graph_Tools

def Cell_Find_From_Graph(
        input_graph,
        find_thres = 2.5,
        max_pix = 1000,
        min_pix = 20,
        shape_boulder = [20,20,20,20], 
        sharp_gauss = ([7,7],1.5),
        back_gauss = ([15,15],7),
        size_limit = 20
        ):
    """
    Find Cell From Graph,this graph can be average intensity or On-Off sub map.

    Parameters
    ----------
    input_graph : (2D Array)
        Input graph. Use this graph to find cell.
    find_thres : (float), optional
        How many std we use in finding cells. The default is 2.5.
    max_pix : (int), optional
        Max cell area. Bigger than this will be ignored. The default is 1000.
    min_pix : (int), optional
        Mininum cell area. Smaller than this will be ignored. The default is 20.
    shape_boulder : (4 element list), optional
        Cell find boulder, usually same as align range. The default is 20.
        [Up,Down,Left,Right]
    sharp_gauss : (turple), optional
        Sharp gaussian blur. Cell data will be found there. The default is ([7,7],1.5).
    blur_gauss : (turple), optional
        Coarse gaussian blur, this value determine local average areas. The default is ([15,15],7).
    size_limit : (int), optional
        Width and Length limitarion. Area longer than this will be ignored. The default is 20.

    Returns
    -------
    Cell_Finded : (Dictionary)
        All value will return to this dictionary. Keys are shown below:
            ['All_Cell_Information']: skimage data, contains all cell location and areas.
            ['Cell_Graph']:Cell location graph.
            ['Annotate_Graph']:Annotated cell graph, cell graph with number.
            ['Combine_Graph']:Combine input graph and cell graph, circle every cell.

    """
    # Get Original Cell Graph
    sharp_mask = Calculator.Normalized_2D_Gaussian_Generator(sharp_gauss)
    back_mask = Calculator.Normalized_2D_Gaussian_Generator(back_gauss)
    sharp_blur = scipy.ndimage.correlate(input_graph,sharp_mask,mode = 'reflect').astype('f8')
    back_blur = scipy.ndimage.correlate(input_graph,back_mask,mode = 'reflect').astype('f8')
    im_cell = (sharp_blur-back_blur)/np.max(sharp_blur-back_blur) # Local Max value shown in this graph.
    level = np.mean(im_cell)+find_thres*np.std(im_cell)
    origin_cells = im_cell>level # Original cell graphs, need to be filtered later.
    # Revome boulder.
    origin_cells = Graph_Tools.Boulder_Fill(origin_cells, shape_boulder, 0)
    # Then remove small areas, other removal have no direct function, so we have to do it later.
      
        
    
    
    return origin_cells
    
    #return Cell_Finded
