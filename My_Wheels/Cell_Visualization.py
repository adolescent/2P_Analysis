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

#%% Function 3: Location Annotate.
def Location_Annotation(
        id_lists,
        annotate_size,
        graph_size = (512,512)
        ):
    """
    Annotate point with circle in blank graphs

    Parameters
    ----------
    id_lists : (list)
        List of all ids. every id need to be a turple(y,x).
    annotate_size : (int)
        Radius of annotation circle.
    graph_size:(turple),optional
        (x,y) size of output graph. The default is (512,512)

    Returns
    -------
    annotation_map : (2D array, dtype = 'u1')
        Annotation graph. 
    """
    annotation_map = np.zeros(shape = graph_size,dtype = 'u1')
    for i in range(len(id_lists)):
        x = id_lists[i][1]
        y = id_lists[i][0]
        cv2.circle(annotation_map,(x,y),annotate_size,255,0)
    return annotation_map
#%% Function 4: Part Cell Visialize
def Part_Cell_Visualize(
        cell_id_list,
        cell_information,
        color = [255,255,255],
        mode = 'Fill'
        ):
    """
    Plot part of cell in cell informations.

    Parameters
    ----------
    cell_id_list : (list)
        Cell id list you want to plot.
    cell_information : (list)
        Skimage produced cell information.
    color : (3-element-list), optional
        Color of plot cells. The default is [255,255,255].
    mode : ('Fill' or 'Boulder'), optional
        Type of cell visualization. The default is 'Fill'.

    Returns
    -------
    Visualized_Part_Cells : (2D Array)
        Visualized graph.

    """
    used_cell_information = []
    for i in range(len(cell_id_list)):
        used_cell_information.append(cell_information[cell_id_list[i]])
    Visualized_Part_Cells = Cell_Visualize(used_cell_information,color = color,mode = mode)    
    return Visualized_Part_Cells

#%% Function 5: Compare two cell informations.
def Cell_Information_Compare(cell_finded_A,
                             cell_finded_B,
                             shift_limit = 10,
                             graph_size = (512,512),
                             plot = False,
                             save_folder = '',
                             show_time = 2000
                             ):
    """
    Compare cell find consistency between two cell datas.

    Parameters
    ----------
    cell_finded_A : (dic)
        Cell Find Dictionary A, A is base.
    cell_info_B : (dic)
        Cell Find Dictionary B, usa A to compare B.
    shift_limit : (int),optional
        Max shift pix for two cell to be regarded as one. The default is 10.
    graph_size : (2-element-turple),optional
        Graph size. Pre given for convenient. The default is (512,512).
    plot : (Bool),optional
        If True, all graph will be ploted and save in save_folder. The default is False.
    save_folder : (str),optional
        If plot is True, save_folder shall be given. The default is ''.
    show_time:(int),optional
        Graph show time during plot. The default is 2s.
    Returns
    -------
    Compare_Dictionary : (Dictionary)
        Compare information dictionaries.
    """
    from My_Wheels.Cell_Location_Compare import Cell_Location_Compare
    import My_Wheels.List_Operation_Kit as List_Tools
    import My_Wheels.OS_Tools_Kit as OS_Tools
    # First, generate simple intersection & union map.
    Compare_Dictionary = {}
    intersection,union,area = Graph_Tools.Graph_Overlapping(cell_finded_A['Cell_Graph'],cell_finded_B['Cell_Graph'])
    Compare_Dictionary['Intersection_Map'] = intersection
    Compare_Dictionary['Union_Map'] = union
    Compare_Dictionary['Cell_Areas'] = area
    # Second, get origin compare dictionary.
    cell_info_A = cell_finded_A['All_Cell_Information']
    cell_info_B = cell_finded_B['All_Cell_Information']
    compare_information = Cell_Location_Compare(cell_info_A,cell_info_B,shift_limit = shift_limit)
    Compare_Dictionary['Compare_Information'] = compare_information
    all_coactive_cell_loc = []
    for i in range(len(compare_information)):
        if compare_information[i][0] != -1: # Which means there is a match for cell i in set A.
            current_loc = compare_information[i][1]
            all_coactive_cell_loc.append((int(current_loc[0]),int(current_loc[1])))
    coactive_cell_annotate = Location_Annotation(all_coactive_cell_loc,5,graph_size)
    Compare_Dictionary['Coactive_Cell_Annotate'] = coactive_cell_annotate
    # Last, draw match and unmatch cell seperately.
    A_Set_Cell_Num = len(cell_info_A)
    B_Set_Cell_Num = len(cell_info_B)
    Paired_A_Cell_ids = []
    Paired_B_Cell_ids = []
    for i in range(A_Set_Cell_Num):
        current_compare_B_id = compare_information[i][0]
        if current_compare_B_id != -1:# as a match
            Paired_A_Cell_ids.append(i)
            Paired_B_Cell_ids.append(current_compare_B_id)
    Paired_A_Cell_ids = list(set(Paired_A_Cell_ids))
    Compare_Dictionary['Paired_Cell_Num'] = len(Paired_A_Cell_ids)
    print(str(len(Paired_A_Cell_ids))+' of '+str(A_Set_Cell_Num)+' Cells are paired.')
    Paired_B_Cell_ids = list(set(Paired_B_Cell_ids))
    Unpaired_A_Cell_ids = List_Tools.List_Subtraction(list(range(A_Set_Cell_Num)),Paired_A_Cell_ids)
    Unpaired_B_Cell_ids = List_Tools.List_Subtraction(list(range(B_Set_Cell_Num)),Paired_B_Cell_ids)
    # Then get specific cells with 
    Paired_A_Cell_Graph = Part_Cell_Visualize(Paired_A_Cell_ids,cell_info_A)
    Paired_B_Cell_Graph = Part_Cell_Visualize(Paired_B_Cell_ids,cell_info_B)
    Unpaired_A_Cell_Graph = Part_Cell_Visualize(Unpaired_A_Cell_ids,cell_info_A)
    Unpaired_B_Cell_Graph = Part_Cell_Visualize(Unpaired_B_Cell_ids,cell_info_B)
    # Then put all graphs on one map
    combine_graph = Graph_Tools.Combine_Graphs((Paired_A_Cell_Graph,
                                                Paired_B_Cell_Graph,
                                                Unpaired_A_Cell_Graph,
                                                Unpaired_B_Cell_Graph
                                                ),all_colors = ['c','g','r','b'])
    Compare_Dictionary['Paired_A_Cell_Graph'] = Paired_A_Cell_Graph
    Compare_Dictionary['Paired_B_Cell_Graph'] = Paired_B_Cell_Graph
    Compare_Dictionary['Unpaired_A_Cell_Graph'] = Unpaired_A_Cell_Graph
    Compare_Dictionary['Unpaired_B_Cell_Graph'] = Unpaired_B_Cell_Graph
    Compare_Dictionary['Match_Graph_Combine'] = combine_graph
    # if plot is true, plot all graphs in specific folder.
    if plot == True:
        OS_Tools.Save_Variable(save_folder,'Compare_Matrix',Compare_Dictionary)
        all_keys = list(Compare_Dictionary.keys())
        all_keys.remove('Cell_Areas')
        all_keys.remove('Compare_Information')
        all_keys.remove('Paired_Cell_Num')
        for i in range(8):
            Graph_Tools.Show_Graph(Compare_Dictionary[all_keys[i]],
                                   all_keys[i],
                                   save_path = save_folder,
                                   show_time = show_time)
    return Compare_Dictionary
#%% Test Runs
if __name__ == '__main__':
    from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Graph
    input_graph = cv2.imread(r'E:\ZR\Data_Temp\191215_L77_2P\Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg\Results\Global_Average_After_Align.tif',-1)
    test = Cell_Find_From_Graph(input_graph)['All_Cell_Information']
    visualized_graph = Cell_Visualize(test,color = [0,0,255],mode = 'Fill')
    Graph_Tools.Show_Graph(visualized_graph,'test',save_path = '',write = False)