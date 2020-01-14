# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:47:51 2019

@author: ZR

Graph Operation kits, this tool box aims at doing all graph Works
"""

import cv2
import numpy as np

#%% Function1: Graph Average(From File).

def Average_From_File(Name_List):
    """
    Average Graph Files, return an aligned matrix. RGB Graph shall be able to use it (not tested).

    Parameters
    ----------
    Name_List : (list)
        File Name List. all list units shall be a direct file path.

    Returns
    -------
    averaged_graph : (2D ndarray, float64)
        Return averaged graph, data type f8 for convenient.

    """
    graph_num = len(Name_List)
    temple_graph = cv2.imread(Name_List[0],-1)
    averaged_graph = np.zeros(shape = temple_graph.shape,dtype = 'f8')
    for i in range(graph_num):
        current_graph = cv2.imread(Name_List[i],-1).astype('f8')# Read in graph as origin depth, and change into f8
        averaged_graph += current_graph/graph_num
    return averaged_graph

#%% Function2: Clip And Normalize input graph
def Clip_And_Normalize(input_graph,clip_std = 2.5,normalization = True,bit = 'u2'):
    """
    Clip input graph,then normalize them to specific bit depth, output graph be shown directly.
    
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
    #Step1, clip
    input_graph = input_graph.astype('f8')
    if clip_std > 0:
        lower_level = input_graph.mean()-clip_std*input_graph.std()
        higher_level = input_graph.mean()+clip_std*input_graph.std()
        clipped_graph = np.clip(input_graph,lower_level,higher_level)
    else:
        print('No Clip Done.')
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
    current_bit : ('u1' or 'u2'), optional
        Dtype of output graph. The default is 'u2'.

    Returns
    -------
    output_graph : (2D or 3D Array)
        Output graph.

    """
    graph = graph.astype('f8')
    normalized_graph = (graph-np.min(graph))/(np.max(graph)-np.min(graph))
    if output_bit == 'u1':
        max_value = 255
    elif output_bit == 'u2':
        max_value = 65535
    else:
        raise IOError('Incorrect bit detph')
    output_graph = (normalized_graph*max_value).astype(output_bit)
    
    return output_graph