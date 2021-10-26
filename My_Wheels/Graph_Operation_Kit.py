# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:47:51 2019

@author: ZR

Graph Operation kits, this tool box aims at doing all graph Works
"""

import cv2
import numpy as np
import os
import My_Wheels.Filters as Filter
import My_Wheels.List_Operation_Kit as lt
import My_Wheels.OS_Tools_Kit as ot

#%% Function1: Graph Average(From File).

def Average_From_File(Name_List,LP_Para = False,HP_Para = False,filter_method = False):
    '''
    Average graph from file. filter is allowed.

    Parameters
    ----------
    Name_List : (list)
        File name of all input graph.
    LP_Para : (turple), optional
        Use False to skip. Low pass parameter. The default is False.
    HP_Para : (turple), optional
        Use False to skip. High pass parameter. The default is False.
    filter_method : (str), optional
        Use False to skip. Filter method. The default is False.

    Returns
    -------
    averaged_graph : TYPE
        DESCRIPTION.

    '''
    graph_num = len(Name_List)
    temple_graph = cv2.imread(Name_List[0],-1)
    origin_type = temple_graph.dtype
    averaged_graph = np.zeros(shape = temple_graph.shape,dtype = 'f8')
    for i in range(graph_num):
        current_graph = cv2.imread(Name_List[i],-1).astype('f8')# Read in graph as origin depth, and change into f8
        if filter_method != False: # Meaning we need to filter graph.
            current_graph = Filter.Filter_2D(current_graph,LP_Para,HP_Para,filter_method)
            
        averaged_graph += current_graph/graph_num
    averaged_graph = averaged_graph.astype(origin_type)
    return averaged_graph

#%% Function2: Clip And Normalize input graph
def Clip_And_Normalize(input_graph,clip_std = 2.5,normalization = True,bit = 'u2'):
    """
    Clip input graph,then normalize them to specific bit depth, output graph be shown directly.
    If not normalize, just return origin dtype.
    
    Parameters
    ----------
    input_graph : (2D ndarray)
        Input graph matrix. Need to be a 2D ndarray.
    clip_std : (float), optional
        How much std will the input graph be clipped into. The default is 2.5, holding 99% data unchanged.
        This Variable can be set to -1 to skip clip.
    normalization : (Bool), optional
        Whether normalization is done here. The default is True, if False, no normalization will be done here.
    bit : ('u2','u1','f8'), optional
        dtype of output graph. This parameter will affect normalization width. The default is 'u2'.

    Returns	
    -------
    processed_graph : (2D ndarray)
        Output graphs.

    """
    origin_dtype = str(input_graph.dtype)
    #Step1, clip
    input_graph = input_graph.astype('f8')
    if clip_std > 0:
        lower_level = input_graph.mean()-clip_std*input_graph.std()
        higher_level = input_graph.mean()+clip_std*input_graph.std()
        clipped_graph = np.clip(input_graph,lower_level,higher_level)
    else:
        print('No Clip Done.')
        clipped_graph = input_graph
        
    #Step2, normalization
    norm_graph = (clipped_graph-clipped_graph.min())/(clipped_graph.max()-clipped_graph.min())
    if normalization == True:
        if bit == 'u2':
            processed_graph = (norm_graph*65535).astype('u2')
        elif bit == 'u1':
            processed_graph = (norm_graph*255).astype('u1')
        elif bit == 'f8':
            print('0-1 Normalize data returned')
            processed_graph = norm_graph
        else:
            raise IOError('Output dtype not supported yet.')
    else:
        print('No Normalization Done.')
        processed_graph = clipped_graph.astype(origin_dtype)
        
    return processed_graph
#%% Function3: Show Graph and Write them.
def Show_Graph(input_graph,graph_name,save_path,show_time = 5000,write = True,graph_formation = '.tif'):
    """
    Show input graph, and write them in ordered path.

    Parameters
    ----------
    input_graph : (2D Ndarray,dtype = 'u1' or 'u2')
        Input graph. Must be plotable dtype, or there will be problems in plotting.
    graph_name : (str)
        Graph name.
    save_path : (str)
        Save path. can be empty is write = False
    show_time : (int), optional
        Graph show time, ms. This value can be set to 0 to skip show.The default is = 5000.
    write : (Bool), optional
        Whether graph is written. If false, show graph only. The default is True.
    graph_formation : (str),optional
        What kind of graph you want to save. The default is '.tif'

    Returns
    -------
    None.

    """
    if show_time != 0:
        cv2.imshow(graph_name,input_graph)
        cv2.waitKey(show_time)
        cv2.destroyAllWindows()
    if write == True:
        
        if os.path.exists(save_path):# If path exists
            cv2.imwrite(save_path+r'\\'+graph_name+graph_formation,input_graph)
        else:# Else, creat save folder first.
            os.mkdir(save_path)
            cv2.imwrite(save_path+r'\\'+graph_name+graph_formation,input_graph)
        
#%% Function 4: Graph Boulder Cut
def Graph_Cut(graph,boulders):
    """
    Cut Graph with specific boulders.

    Parameters
    ----------
    graph : (ndarray)
        Input graph.
    boulders : (list,length = 4,element = int)
        4 element list. Telling cut pix of 4 directions.
        [0]:Up; [1]:Down; [2]:Left; [3]:Right

    Returns
    -------
    cutted_graph : (ndarray)
        Cutted graph. dtype consist with input graph.

    """
    ud_range,lr_range = np.shape(graph)
    if ud_range < (boulders[0]+boulders[1]) or lr_range < (boulders[2]+boulders[3]):
        raise IOError('Cut bouder too big, misison impossible.')
    cutted_graph = graph[boulders[0]:(ud_range-boulders[1]),boulders[2]:(lr_range-boulders[3])]
    
    return cutted_graph
#%% Function 5 Boulder Fill
def Boulder_Fill(graph,boulders,fill_value):
    """
    Fill Graph Boulder with specific value. Graph shape will not change.

    Parameters
    ----------
    graph : (2D Array)
        Input graph.
    boulders : (4 element list)
        Telling boulder width in all 4 directions.
        [0]:Up; [1]:Down; [2]:Left; [3]:Right
        fill_value : (number)
        Value you want to fill in boulders.

    Returns
    -------
    graph : (2D Array)
        Boulder filled graph.

    """
    length,width = np.shape(graph)
    graph[0:boulders[0],:] = fill_value
    graph[(length-boulders[1]):length,:] = fill_value
    graph[:,0:boulders[2]] = fill_value
    graph[:,(width-boulders[3]):width] = fill_value
    
    return graph
#%% Function 6 Grap_Combine
def Graph_Combine(graph_A,graph_B,bit = 'u1'):
    """
    Combine 2 input graphs, just add together.

    Parameters
    ----------
    graph_A : (Input Graph, 2D or 3D Array)
        Graph A. Gray map 2D, color map 3D.
    graph_B : (Input Graph, 2D or 3D Array)
        Graph B. same type as A.
    bit : ('u1' or 'u2'), optional
        DESCRIPTION. The default is 'u1'.

    Returns
    -------
    combined_graph : TYPE
        Graph same shape as input.

    """
    graph_A = graph_A.astype('f8')
    graph_B = graph_B.astype('f8')
    # Check graph shape
    if np.shape(graph_A) != np.shape(graph_B):
        raise IOError('Graph Shape not match, CHECK please.')
    # Determine max pix value.
    if bit == 'u1':
        max_value = 255
    elif bit == 'u2':
        max_value = 65535
    else:
        raise IOError('Incorrect bit depth.')
    # Then Add Up 2 graphs, then clip them.
    combined_graph = np.clip(graph_A + graph_B,0,max_value).astype(bit)
    
    return combined_graph
#%% Function7 Graph Depth Change
def Graph_Depth_Change(graph,output_bit = 'u2'):
    """
    Change Graph Depth between uint8 and uint16. Change from 1 to another.

    Parameters
    ----------
    graph : (Input Graph, 2D or 3D Array)
        Input Graph of .
    output_bit : ('u1' or 'u2'), optional
        Dtype of output graph. The default is 'u2'.

    Returns
    -------
    output_graph : (2D or 3D Array)
        Output graph.

    """
    graph = graph.astype('f8')
    change_index = 65535/255
    if output_bit == 'u1': # Which means we will change 16bit to 8 bit
        output_graph = (graph/change_index).astype('u1')
    elif output_bit == 'u2': # Which means change 8 bit to 16 bit
        output_graph = (graph*change_index).astype('u2')
    
    
# =============================================================================
#     normalized_graph = (graph-np.min(graph))/(np.max(graph)-np.min(graph))
#     if output_bit == 'u1':
#         max_value = 255
#     elif output_bit == 'u2':
#         max_value = 65535
#     else:
#         raise IOError('Incorrect bit detph')
#     output_graph = (normalized_graph*max_value).astype(output_bit)
# =============================================================================
    
    return output_graph
#%% Function8 Sub Graph Generator
def Graph_Subtractor(tif_names,A_Sets,B_Sets,clip_std = 2.5,output_type = 'u2'):
    """
    Get A-B graph.

    Parameters
    ----------
    tif_names : (list)
        List of all aligned tif names.This can be found in Align_Property(Cross_Run_Align.Do_Align())
    A_Sets : (list)
        Frame id of set A.Use this to get specific frame name.
    B_Sets : (list)
        Frame id of set B.
    clip_std : (number), optional
        How many std will be used to clip output graph. The default is 2.5.
    output_type : ('f8','u1','u2'), optional
        Data type of output grpah. The default is 'u2'.

    Returns
    -------
    subtracted_graph : (2D ndarray)
        Subtracted graph.
    dF_F:(float)
        Rate of changes, (A-B)/B

    """
    A_Set_tif_names = []
    for i in range(len(A_Sets)):
        A_Set_tif_names.append(tif_names[A_Sets[i]])
    B_Set_tif_names = []
    for i in range(len(B_Sets)):
        B_Set_tif_names.append(tif_names[B_Sets[i]])
    A_Set_Average = Average_From_File(A_Set_tif_names)
    B_Set_Average = Average_From_File(B_Set_tif_names)
    simple_sub = A_Set_Average - B_Set_Average
    dF_F = simple_sub.mean()/B_Set_Average.mean()
    subtracted_graph = Clip_And_Normalize(simple_sub,clip_std,bit = output_type)
    return subtracted_graph,dF_F
    
#%% Function 9 Graph Overlapping/Union
def Graph_Overlapping(graph_A,graph_B,thres = 0.5):
    """
    Simple overlapping calculation, used to compare cell locations

    Parameters
    ----------
    graph_A : (2D Array/2D Array*3)
        Graph A, regarded as gray graph.
    graph_B : (2D Array/2D Array*3)
        Graph B, regarded as gray graph.
    thres : (0~1 float)
        Threshold used for binary data.

    Returns
    -------
    intersection_graph : (2D Array, dtype = 'u1')
        Overlapping graph. Both active point will be shown on this graph.
    union_graph : (2D Array, dtype = 'u1')
        Union graph. All active point in single graph will be shown.
    active_areas:(list,3-element)
        [A_areas,B_areas,intersection_areas]
    """
    # Process graph first.
    A_shape = np.shape(graph_A)
    if len(A_shape) == 3:# Meaning color graph.
        used_A_graph = cv2.cvtColor(graph_A,cv2.COLOR_BGR2GRAY).astype('f8')
    else:
        used_A_graph = graph_A.astype('f8')
    norm_A_graph = (used_A_graph-used_A_graph.min())/(used_A_graph.max()-used_A_graph.min())
    bool_A = norm_A_graph > thres
    B_shape = np.shape(graph_B)
    if len(B_shape) == 3:# Meaning color graph.
        used_B_graph = cv2.cvtColor(graph_B,cv2.COLOR_BGR2GRAY).astype('f8')
    else:
        used_B_graph = graph_B.astype('f8')
    norm_B_graph = (used_B_graph-used_B_graph.min())/(used_B_graph.max()-used_B_graph.min())
    bool_B = norm_B_graph > thres    
    # Then calculate intersection and union.
    bool_intersection = bool_A*bool_B
    bool_union = bool_A+bool_B
    A_areas = bool_A.sum()
    B_areas = bool_B.sum()
    intersection_areas = bool_intersection.sum()
    active_areas = [A_areas,B_areas,intersection_areas]
    #return bool_intersection,bool_union
    # Output as u1 type at last.
    intersection_graph = (bool_intersection.astype('u1'))*255
    union_graph = (bool_union.astype('u1'))*255
    return intersection_graph,union_graph,active_areas
    
#%% Functino 10 : Plot several input graph on same map, using different colors.
from My_Wheels.Calculation_Functions import Color_Dictionary
def Combine_Graphs(
        graph_turples,
        all_colors = ['r','g','b','y','c','p'],
        graph_size = (512,512)
        ):
    """
    Combine several map together, using different colors. This function better be used on binary graphs.

    Parameters
    ----------
    graph_turples : (turple)
        Turple list of input graph, every element shall be a 2D array, 6 map supported for now.
    all_colors : (list), optional
        Sequence of input graph colors. The default is ['r','g','b','y','c','p'].
    graph_size : (2-element turple),optional
        Graph size. Pre defined for convenience

    Returns
    -------
    Combined_Map : (2D array, 3 channel,dtype = 'u1')
        Combined graph. Use u1 graph.
    """
    Combined_Map = np.zeros(shape = (graph_size[0],graph_size[1],3),dtype = 'f8')
    Graph_Num= len(graph_turples)
    if Graph_Num > len(all_colors):
        raise ValueError('Unable to combine so many graphs for now..\n')
    for i in range(Graph_Num):
        current_graph = graph_turples[i]
        if len(np.shape(current_graph)) == 3:# for color map,change into gray.
            current_graph = cv2.cvtColor(current_graph,cv2.COLOR_BGR2GRAY).astype('f8')
        current_graph = Clip_And_Normalize(current_graph,clip_std = 0, bit = 'f8')# Normalize and to 0-1
        # After processing, draw current graph onto combined map.
        current_color = Color_Dictionary[all_colors[i]]
        Combined_Map[:,:,0] = Combined_Map[:,:,0]+current_graph*current_color[0]
        Combined_Map[:,:,1] = Combined_Map[:,:,1]+current_graph*current_color[1]
        Combined_Map[:,:,2] = Combined_Map[:,:,2]+current_graph*current_color[2]
    # After that, clip output graph.
    Combined_Map = np.clip(Combined_Map,0,255).astype('u1')    
    return Combined_Map
#%% Function 11 : Easy Plot.
def EZPlot(input_graph,show_time = 7000):
    """
    Easy plot, do nothing and just show graph.

    Parameters
    ----------
    input_graph : (2D Array, u1 or u2 dtype)
        Input graph.
    show_time : (int),optional
        Show time. The default is 7s.
    Returns
    -------
    int
        Fill in blank.

    """
    graph_name = 'current_graph'
    Show_Graph(input_graph,graph_name,save_path = '',show_time = show_time,write = False)
    return 0

#%% Function 12 : Easy Clip.
def EZClip(input_graph,clip_std = 5,max_lim = 'inner',min_lim = 'inner'):
    """
    Easy Clip, clip graph with given std. Return same bit depth as input.

    Parameters
    ----------
    input_graph : (2D_NdArray)
        Graph need to be clip.
    clip_std : (float),optional
        Std of clip. The default is 5
    max_lim : (float),optional.
        If number is given, use this instead of std max.
    min_lim : (float),optional.
        If number is given, use this instead of std min.

    Returns
    -------
    clipped_graph : (2D_NdArray)
        Clipped graph. Bit depth same as input.

    """
    origin_type = str(input_graph.dtype)
    raw_graph = input_graph.astype('f8') # Turn into float64 for calculation.
    mean = raw_graph.mean()
    std = raw_graph.std()
    upper_lim = mean+std*clip_std
    lower_lim = mean-std*clip_std
    if origin_type == 'uint8':
        lower_lim = max(lower_lim,0)
        upper_lim = min(upper_lim,255)
    elif origin_type == 'uint16':
        lower_lim = max(lower_lim,0)
        upper_lim = min(upper_lim,65535)
        
    if max_lim != 'inner':
        upper_lim = max_lim
    if min_lim != 'inner':
        lower_lim = min_lim
        
    clipped_graph = np.clip(raw_graph,lower_lim,upper_lim).astype(origin_type)
    
    
    return clipped_graph

#%% Function 13 : Easy Normalization
def EZNormalize(input_graph):
    '''
    Normalization Tools, return 0-1 matrix. Usable to all 

    Parameters
    ----------
    input_graph : (NdArray)
        Input graph. Any dtype is okay.

    Returns
    -------
    normalized_graph : (NdArray)
        Normalized graph. Data type is 0-1 ranged float64.

    '''
    origin_graph = input_graph.astype('f8')
    max_value = origin_graph.max()
    min_value = origin_graph.min()
    normalized_graph = (origin_graph-min_value)/(max_value-min_value)
    return normalized_graph

#%% Function 14 : Graph Central Caltulator
def Graph_Center_Calculator(input_graph,center_mode = 'weight',annotate_brightness = 50):
    """
    Calculate graph center, return coordinate and annotate graph.

    Parameters
    ----------
    input_graph : (2D Nd-Array)
        Input graph.
    center_mode : ('weight' or 'binary')
        Mode of center find. 'weight' means intense of point contained in calculation, 'binary' center calculated from thresholed graph.        
    annotate_brightness : (int), optional
        0-100, brightness of graph in annotate graph. The default is 50.

    Returns
    -------
    center_loc : (turple)
        coordinate of mass center YX sequence.
    annotate_graph : (2D_NdArray)
        Graph of annotation. brightness above.

    """
    from skimage import filters
    from skimage.measure import regionprops
    threshold_value = filters.threshold_otsu(input_graph)
    labeled_foreground = (input_graph > threshold_value).astype(int)# 1/0 boulder map.
    properties = regionprops(labeled_foreground, input_graph)
    if center_mode == 'weight':
        center_loc = properties[0].weighted_centroid
    elif center_mode == 'binary':
        center_loc = properties[0].centroid
    else:
        raise ValueError('Invalid center method.')
    # Then draw annotation map.
    annotate_graph = EZNormalize(input_graph)*annotate_brightness/100
    y_loc = int(center_loc[0])
    x_loc = int(center_loc[1])
    annotate_graph[y_loc-2:y_loc+2,x_loc-2:x_loc+2]=1
    annotate_graph = (annotate_graph*255).astype('u1')
    return center_loc,annotate_graph

#%% Function 15: Graph Boulder Extend
def Boulder_Extend(graph,pix_extend,fill_value = 0):
    '''
    Extend graph boulders with specific value. Graph shape WILL change.

    Parameters
    ----------
    graph : (2D Array)
        Input graph.
    pix_extend : (list)
        [Up,Down,Left,Right]. List of pix num we extend.
    fill_value : (int), optional
        The value we fill into the extend pix. The default is 0.

    Returns
    -------
    extended_graph : (2D Array)
        Extended graph. Shape shall be checked.

    '''
    height,width = graph.shape
    origin_dtype = graph.dtype
    extend_height = height+pix_extend[0]+pix_extend[1]
    extend_width = width+pix_extend[2]+pix_extend[3]
    extended_graph = np.ones(shape = (extend_height,extend_width),dtype = origin_dtype)*fill_value
    extended_graph[pix_extend[0]:(pix_extend[0]+height),pix_extend[2]:(pix_extend[2]+width)] = graph
    
    return extended_graph
#%% Function16, graph rotation.
def Graph_Twister(origin_graph,angle):
    '''
    Rotate graph in given angle. Fill boulder with white. Clockwise.

    Parameters
    ----------
    origin_graph : (2D Array).
        Input graph. Can be color or grayscale. Output will be the same format.
    angle : (int)
        Clockwise rotation angle. Degree system.

    Returns
    -------
    twisted_graph : (2D Array)
        Twisted graph. Boulder will be extended to fullfill the graph. Fill white in the blank.

    '''
    h,w = origin_graph.shape[:2]# first 2 axis.
    (cx,cy) = (w/2,h/2)
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    twisted_graph = cv2.warpAffine(origin_graph,M,(nW,nH),borderValue=(0,0,0))
    return twisted_graph
#%% Function 17, Global average generator
def Global_Averagor(all_folder_list,
                    sub_folders = r'\Results\Affined_Frames'
                    ):
    '''
    Average global graph from all subgraphs.

    Parameters
    ----------
    all_folder_list : TYPE
        DESCRIPTION.
    sub_folders : TYPE, optional
        DESCRIPTION. The default is r'\Results\Affined_Frames'.

    Returns
    -------
    global_averaged_graph : TYPE
        DESCRIPTION.

    '''
    all_folders = lt.List_Annex(all_folder_list, [sub_folders])
    all_tif_name = []
    for i in range(len(all_folders)):
        current_tif_name = ot.Get_File_Name(all_folders[i])
        all_tif_name.extend(current_tif_name)
    global_averaged_graph = Average_From_File(all_tif_name)
    
    return global_averaged_graph
    

