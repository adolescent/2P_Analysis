# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:53:39 2020

@author: ZR
"""
#%%
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import cv2
import numpy as np
import My_Wheels.Filters as Filters
from tqdm import tqdm
#%%
def Video_From_File(
        data_folder,
        plot_range = (0,9999),
        graph_size = (472,472),
        file_type = '.tif',
        fps = 15,
        gain = 20,
        LP_Gaussian = ([5,5],1.5),
        frame_annotate = True,
        cut_boulder = [20,20,20,20]
        ):
    '''
    Write all files in a folder as a video.

    Parameters
    ----------
    data_folder : (std)
        Frame folder. All frame in this folder will be write into video. Dtype shall be u2 or there will be a problem.
    graph_size : (2-element-turple), optional
        Frame size AFTER cut. The default is (472,472).
    file_type : (str), optional
        Data type of graph file. The default is '.tif'.
    fps : (int), optional
        Frame per second. The default is 15.
    gain : (int), optional
        Show gain. The default is 20.
    LP_Gaussian : (turple), optional
        LP Gaussian Filter parameter. Only do low pass. The default is ([5,5],1.5).
    frame_annotate : TYPE, optional
        Whether we annotate frame number on it. The default is True.
    cut_boulder : TYPE, optional
        Boulder cut of graphs, UDLR. The default is [20,20,20,20].


    Returns
    -------
    bool
        True if function processed.

    '''

    all_tif_name = OS_Tools.Get_File_Name(path = data_folder,file_type = file_type)
    start_frame = plot_range[0]
    end_frame = min(plot_range[1],len(all_tif_name))
    all_tif_name = all_tif_name[start_frame:end_frame]
    graph_num = len(all_tif_name)
    video_writer = cv2.VideoWriter(data_folder+r'\\Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
    #video_writer = cv2.VideoWriter(data_folder+r'\\Video.avi',-1,fps,graph_size,0)
    for i in tqdm(range(graph_num)):
        raw_graph = cv2.imread(all_tif_name[i],-1).astype('f8')
        # Cut graph boulder.
        raw_graph = Graph_Tools.Graph_Cut(raw_graph, cut_boulder)
        # Do gain then
        gained_graph = np.clip(raw_graph.astype('f8')*gain/256,0,255).astype('u1')
        # Then do filter, then 
        if LP_Gaussian != False:
            u1_writable_graph = Filters.Filter_2D(gained_graph,LP_Gaussian,False)
        else:
            u1_writable_graph = gained_graph.astype('f8')
        if frame_annotate == True:
            cv2.putText(u1_writable_graph,'Stim ID = '+str(i),(250,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255),1)
        video_writer.write(u1_writable_graph)
    del video_writer 
    return True

#%%
def Video_From_mat(input_matrix,
        save_path,
        fps = 4,
        frame_annotate = True
        ):

    print('Generate video from file, make sure input file is u1 type.')
    graph_size = (input_matrix.shape[1],input_matrix.shape[0])
    graph_num = input_matrix.shape[2]
    video_writer = cv2.VideoWriter(save_path+r'\\Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
    for i in tqdm(range(graph_num)):
        c_graph = input_matrix[:,:,i]
        u1_writable_graph = c_graph.astype('f8')
        if frame_annotate == True:
            cv2.putText(u1_writable_graph,'Stim ID = '+str(i),(250,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255),1)
        u1_writable_graph = u1_writable_graph.astype('u1')
        video_writer.write(u1_writable_graph)
    del video_writer 

    return True
