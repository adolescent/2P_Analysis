# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:53:39 2020

@author: ZR
"""
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import cv2
import numpy as np
import My_Wheels.Filters as Filters

def Video_From_File(
        data_folder,
        graph_size = (512,512),
        file_type = '.tif',
        fps = 15,
        gain = 20,
        LP_Gaussian = ([5,5],1.5),
        frame_annotate = True,
        cut_boulder = [20,20,20,20],
        ):
    '''
    Write all files in a folder as a video.

    Parameters
    ----------
    data_folder : TYPE
        DESCRIPTION.
    graph_size : TYPE, optional
        DESCRIPTION. The default is (512,512).
    file_type : TYPE, optional
        DESCRIPTION. The default is '.tif'.
    fps : TYPE, optional
        DESCRIPTION. The default is 15.
    gain : TYPE, optional
        DESCRIPTION. The default is 20.
    LP_Gaussian : TYPE, optional
        DESCRIPTION. The default is ([5,5],1.5).
    frame_annotate : TYPE, optional
        DESCRIPTION. The default is True.
    cut_boulder : TYPE, optional
        DESCRIPTION. The default is [20,20,20,20].
     : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    '''

    all_tif_name = OS_Tools.Get_File_Name(path = data_folder,file_type = file_type)
    graph_num = len(all_tif_name)
    video_writer = cv2.VideoWriter(data_folder+r'\\Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
    #video_writer = cv2.VideoWriter(data_folder+r'\\Video.avi',-1,fps,graph_size,0)
    for i in range(graph_num):
        raw_graph = cv2.imread(all_tif_name[i],-1).astype('f8')
        # Cut graph boulder.
        raw_graph = Graph_Tools.Graph_Cut(raw_graph, cut_boulder)
        # Do gain then
        gained_graph = np.clip(raw_graph.astype('f8')*gain/256,0,255).astype('u1')
        # Then do filter, then 
        if LP_Gaussian != False:
            u1_writable_graph = Filters.Filter_2D(gained_graph,LP_Gaussian,False)
        if frame_annotate == True:
            cv2.putText(u1_writable_graph,'Stim ID = '+str(i),(300,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255),1)
        video_writer.write(u1_writable_graph)
    del video_writer 
    return True