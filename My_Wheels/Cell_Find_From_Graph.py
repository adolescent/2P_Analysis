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
import skimage.morphology
import skimage.measure
import Cell_Visualization as Visualize
import cv2


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
            ['Annotate_Cell_Graph']:Annotated cell graph, cell graph with number.
            ['Combine_Graph']:Circle cell on input graph.
            ['Combine_Graph_With_Number']:Circle Cell on input graph, and annotate cell number on input graph.

    """
    # Get Original Cell Graph
    Cell_Finded = {}
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
    cell_washed = skimage.morphology.remove_small_objects(origin_cells,min_pix,connectivity = 1)
    # Get Cell Group Description here.
    cell_label = skimage.measure.label(cell_washed)
    All_Cells = skimage.measure.regionprops(cell_label)
    
    # Then, wash bigger and oversize cell
    for i in range(len(All_Cells)-1,-1,-1): # opposite sequence to avoid id change of pop.
        current_cell = All_Cells[i]
        area = current_cell.convex_area # Cell areas
        # wash too big cell
        if area > max_pix:
            All_Cells.pop(i)
        # wash oversize cell
        else:
            size = np.shape(All_Cells[i].image)
            if np.max(size)>size_limit:
                All_Cells.pop(i)
    # Till now, Cell find is done, Graph Show shall be done here. 
    Cell_Finded['All_Cell_Information'] = All_Cells
    Cell_Finded['Cell_Graph'] = Visualize.Cell_Visualize(All_Cells)
    Cell_Finded['Annotate_Cell_Graph'] = Visualize.Label_Cell(Cell_Finded['Cell_Graph'],All_Cells)
    # Then draw cell on input graph.
    circled_cell = Visualize.Cell_Visualize(All_Cells,color = [0,255,100],mode = 'Boulder')
    input_8bit = Graph_Tools.Graph_Depth_Change(input_graph,output_bit = 'u1')
    input_8bit = cv2.cvtColor(input_8bit,cv2.COLOR_GRAY2RGB)
    Cell_Finded['Combine_Graph'] = Graph_Tools.Graph_Combine(input_8bit,circled_cell)
    Cell_Finded['Combine_Graph_With_Number'] = Visualize.Label_Cell(Cell_Finded['Combine_Graph'], All_Cells)
    
    return Cell_Finded
    
