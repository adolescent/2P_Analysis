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
import My_Wheels.OS_Tools_Kit as OS_Tools
from skimage import filters
import My_Wheels.Graph_Selector as Graph_Selector

#%% Main Function: Return finded graphs.
def Cell_Find_From_Graph(
        input_graph,
        find_thres = 'otsu',
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
    find_thres : (float or 'otsu'), optional
        How many std we use in finding cells. The default is 'otsu', meaning we use ootsu method to find threshold here.
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
    if find_thres == 'otsu':# Add otsu method here.
        level = filters.threshold_otsu(im_cell)
    else:
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
    
#%% Function 2: Cell Find and plot.
def Cell_Find_And_Plot(
        graph_folder,
        graph_name,
        Cell_Label,
        find_thres = 2.5,
        max_pix = 1000,
        min_pix = 20,
        shape_boulder = [20,20,20,20], 
        sharp_gauss = ([7,7],1.5),
        back_gauss = ([15,15],7),
        size_limit = 20    
        ):
    """
    Cell find from file.

    Parameters
    ----------
    graph_folder : (str)
        Graph folder.
    graph_name : (str)
        Graph name. Extend name shall be contained.
    Cell_Label : (str)
        Save sub Folder. Cell data and cell graphs will be saved in this sub folder.
    find_thres,max_pix,min_pix,shape_boulder,sharp_gauss,back_gauss,size_limit : As Function 1, optional
        As Function 1.

    Returns
    -------
    Cell_Finded : TYPE
        DESCRIPTION.

    """
    Base_Graph = cv2.imread(graph_folder + r'\\' + graph_name,-1)
    graph_save_folder = graph_folder + r'\\' + Cell_Label
    Finded_Cells = Cell_Find_From_Graph(Base_Graph,find_thres,max_pix,min_pix,shape_boulder,sharp_gauss,back_gauss,size_limit)
    OS_Tools.Save_Variable(graph_save_folder,Cell_Label,Finded_Cells,extend_name = '.cell')
    all_keys = list(Finded_Cells.keys())
    all_keys.remove('All_Cell_Information')
    for i in range(len(all_keys)):
        Graph_Tools.Show_Graph(Finded_Cells[all_keys[i]],graph_name = all_keys[i],save_path = graph_save_folder,show_time = 2000,write = True)
    return Finded_Cells

#%% Function3, On-Off cell graph generator.
from My_Wheels.Standard_Stim_Processor import Single_Subgraph_Generator
def On_Off_Cell_Finder(
        all_tif_name,
        Stim_Frame_Dic,
        find_thres = 1.5,
        max_pix = 1000,
        min_pix = 20,
        filter_method = 'Gaussian',
        LP_Para = ((5,5),1.5),
        HP_Para = False,
        shape_boulder = [20,20,20,20], 
        sharp_gauss = ([7,7],1.5),
        back_gauss = ([15,15],7),
        size_limit = 20
        ):
    # Generater On-Off graph.
    off_list = Stim_Frame_Dic[0]
    all_keys = list(Stim_Frame_Dic.keys())
    all_keys.remove('Original_Stim_Train')
    all_keys.remove(-1)
    all_keys.remove(0)
    on_list = []
    for i in range(len(all_keys)):
        on_list.extend(Stim_Frame_Dic[all_keys[i]])
    on_off_graph,_,_ = Single_Subgraph_Generator(all_tif_name, on_list, off_list,filter_method,LP_Para,HP_Para,t_map = False)
    on_off_graph = Graph_Tools.Clip_And_Normalize(on_off_graph,clip_std = 2.5)
    Finded_Cells = Cell_Find_From_Graph(on_off_graph,find_thres,max_pix,min_pix,shape_boulder,sharp_gauss,back_gauss,size_limit)
    return on_off_graph,Finded_Cells
#%% Function4, Cell Find from a.i. active .
def Cell_Find_From_active(
        aligned_data_folder,
        find_thres = 1,
        active_mode = 'biggest',
        propotion = 0.05,
        max_pix = 1000,
        min_pix = 20,
        shape_boulder = [20,20,20,20], 
        sharp_gauss = ([7,7],1.5),
        back_gauss = ([15,15],7),
        size_limit = 20  
        ):
    intensity_selected_graph,_ = Graph_Selector.Intensity_Selector(aligned_data_folder,mode = active_mode,propotion=propotion,list_write=False)
    save_folder = '\\'.join(aligned_data_folder.split('\\')[:-1])
    Graph_Tools.Show_Graph(Graph_Tools.Clip_And_Normalize(intensity_selected_graph,clip_std = 5), 'Cell_Find_Base', save_folder)
    Cell_Finded = Cell_Find_And_Plot(save_folder, 'Cell_Find_Base.tif', 'Active_Cell',find_thres,max_pix,min_pix,shape_boulder,sharp_gauss,back_gauss,size_limit)
    return Cell_Finded