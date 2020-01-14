# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:17:56 2020

@author: ZR

Several useful tools for cell visualization.

"""
import numpy as np
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import cv2
#%% Function 1: Visualize Cell
def Cell_Visualize(cell_information,color = [255,255,255],mode = 'Fill'):
    """
    Visualize Cell Locations. Only visualizaion, no weight contained

    Parameters
    ----------
    cell_information : (list)
        Calculated Cell Information. dtype = skimage.morphology data
    color : (3 element list), optional
        RGB value of graph. Remember, cv2 calcutaion use BGR, and you only need to print RGB here. The default is [255,255,255].
    mode : ('Fill' or 'Boulder'), optional
        Return filled cell graph of just circle boulder. The default is 'Fill'.

    Returns
    -------
    visualized_cell_graph : (2D Array)
        Visualized cell graph.

    """
    # Initialize graph
    graph_size = np.shape(cell_information[0]._label_image)+(3,) # Base Graph Size, *3 to Use RGB.
    visualized_cell_graph = np.zeros(graph_size,dtype = 'u1')
    color[2],color[0] = color[0],color[2] # Change RGB into BGR. After switch, color list can be used directly into cv2.
    # Then Cycle Cells
    weight_matrix = np.zeros(np.shape(cell_information[0]._label_image),dtype = 'f8')
    for i in range(len(cell_information)):
        current_cell = cell_information[i]
        y_list,x_list = current_cell.coords[:,0],current_cell.coords[:,1]
        weight_matrix[y_list,x_list] = 1
    # if needed, use boulder recognition.
    if mode == 'Boulder':
        canny = cv2.Canny((weight_matrix*255).astype('u1'),50,150)
        weight_matrix = canny.astype('f8')/255 # Transfer weight matrix into 1/0 matrix
    # Then Colorize Graph.
    for i in range(3):
        visualized_cell_graph[:,:,i] = weight_matrix*color[i]
    
    return visualized_cell_graph

#%% Function2: Label Cell Number.
def Label_Cell(graph,cell_information,color = (0,255,100),font_size = 11):
    """
    Label Cell ID on input graph. Only Numbers.

    Parameters
    ----------
    graph : (3D Ndarray,shape = (size,3))
        Basic BGR Graph. Label Cell on this graph.
    cell_information : (list)
        Skimage Calculated cell informations.
    color : (3 element turple), optional
        Color.Sequence is RGB. The default is (0,255,100).

    Returns
    -------
    Labeled_Graph : (3D Ndarray,shape = (size,3))
        Labled Graph. Cell number will be shown on it.

    """
    from PIL import ImageFont
    from PIL import Image
    from PIL import ImageDraw
    
    font = ImageFont.truetype('arial.ttf',font_size)
    im = Image.fromarray(graph)
    for i in range(len(cell_information)):
        y,x = cell_information[i].centroid
        draw = ImageDraw.Draw(im)
        draw.text((x+5,y+5),str(i),color,font = font,align = 'center')
    Labeled_Graph = np.array(im)
    
    return Labeled_Graph

#%% Test Runs
if __name__ == '__main__':
    from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Graph
    input_graph = cv2.imread(r'E:\ZR\Data_Temp\191215_L77_2P\Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg\Results\Global_Average_After_Align.tif',-1)
    test = Cell_Find_From_Graph(input_graph)['All_Cell_Information']
    visualized_graph = Cell_Visualize(test,color = [0,0,255],mode = 'Fill')
    Graph_Tools.Show_Graph(visualized_graph,'test',save_path = '',write = False)