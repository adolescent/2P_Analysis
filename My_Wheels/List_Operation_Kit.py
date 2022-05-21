# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:10:38 2019

@author: ZR

This Module is Used to do List File Operations, making such work easier.

"""
import numpy as np
#%% Function1:List Annex
def List_Annex( A , B , dilimit = '\\' ):
    
    """
    This Function is used to Annex two part of strings, add same head or tail, useful at path cycle.
    
    Parameters
    ----------
    A : (list)
        Former parts of string,always at the front, can be single or multi units.\n
    B : (list)
        Latter parts of string,always at the front, can be single or multi units.\n
    dilimit : (str,optional)
        Dilimitor of annexed strings,usually use '\' for folder names.
        

    Returns
    -------
    out_str : (list)
        Annexed strings, the same lenth as input.
    
    """

    
    # Check list size, if both A&B have multi unit, raise error.
    if len(A)>1 and len(B)>1:
        raise IOError('Annex method not understand, check strings.')
    elif(len(A)==1): #If have same head
        out_str = []
        for i in range(len(B)):
            out_str.append(str(A[0])+dilimit+str(B[i]))
    else: # If want same tail
        out_str = []
        for i in range(len(A)):
            out_str.append(str(A[i])+dilimit+str(B[0]))
            
    out_str = np.array(out_str)
    return out_str
        
#%% Function 2: List Extend & Cut
def List_extend(input_list,front,tail):
    """
    extend or cut list length.If extend, boulder value will be used.

    Parameters
    ----------
    input_list : (list)
        Input list. All element shall be number.
    front : (int)
        Length want to extend in the front. Negative number will cut list.
    tail : (int)
        Length want to extend at last. Negative number will cut list.

    Returns
    -------
    extended_list : (list)
        Cutted list.

    """
    front_element = input_list[0] # First element at front
    last_element = input_list[-1] # Last element at last
    # Process front first.
    if front >0:
        processing_list = [front_element]*front
        processing_list.extend(input_list)
    else:
        processing_list = input_list[abs(front):]
    # Then process tail parts.    
    if tail > 0:
        tail_list = [last_element]*tail
        processing_list.extend(tail_list)
    elif tail == 0:
        pass
    else:
        processing_list = processing_list[:tail]
    extended_list = processing_list

    return extended_list
#%% Function 3: List To Dictionary
def List_To_Dic(input_list):
    """
    Use list set as key, location of list as value.

    Parameters
    ----------
    input_list : (list)
        All element need to be number.

    Returns
    -------
    Dic : (Dictionary)
        Location of different keys.

    """
    Dic = {}
    for i in range(len(input_list)):
        if input_list[i] in Dic: # if key exists, just append location
            Dic[input_list[i]].append(i)
        else:# If first appearance, creat new key.
            Dic[input_list[i]] = [i]
    return Dic

#%% Function 4: List Subtraction
def List_Subtraction(list_A,list_B):
    """
    A-B. Attention: repeat element in list may cause trouble. Sequence is not considered here.

    Parameters
    ----------
    list_A : (list)
        Mother list.
    list_B : (list)
        Subtracted list. All element need to be in list A.

    Returns
    -------
    subtracted_list : (list)
        All element in A but not in B.
    """
    subtracted_list = list_A.copy()
    if len(set(list_A)) != len(list_A):
        raise IOError('Repeat element in lists, check please.')
    for i in range(len(list_B)):
        subtracted_list.remove(list_B[i])
    return subtracted_list

#%% Function 5: Nearest number find.
def Find_Nearest(input_list,target_number):
    """
    Return nearest number in input_list with target.

    Parameters
    ----------
    input_list : (list)
        All element shall be number, and no repeat is allowed.
    target_number : (float)
        Number you want to match.

    Returns
    -------
    nearest_num : (float)
        Nearest number in input list with target.
    num_loc : (int)
        Location ID of number above.
    """
    input_list = np.asarray(input_list)
    dist = abs(input_list-target_number)
    num_loc = np.where(dist == dist.min())[0][0]
    nearest_num = input_list[num_loc]
    return nearest_num,num_loc

def List_Slicer(input_list,ids_list):
    sliced_list = []
    for i in range(len(ids_list)):
        sliced_list.append(input_list[i])
    return sliced_list
#%% Function 6: Run name getter of 2p
def Run_Name_Producer_2P(run_id_lists):
    subfolder_lists = []
    for i in range(len(run_id_lists)):
        current_runid = str(run_id_lists[i])
        if len(current_runid) == 1:
            subfolder_lists.append('1-00'+current_runid)
        elif len(current_runid) == 2:
            subfolder_lists.append('1-0'+current_runid)
        elif len(current_runid) == 3:
            subfolder_lists.append('1-'+current_runid)
        else:
            raise IOError('Run number impossible, check input plz.')
    return subfolder_lists

#%% Function 7: List element same length
def Element_Same_length(input_list,start = 'head'):
    '''
    cut each element in input lists, make sure they have same length.

    Parameters
    ----------
    input_list : (list)
        Element shall be list too, or this is nonsense.
    start : ('head' or 'tail'), optional
        If length not equal, which direction to reserve. The default is 'head'.

    Returns
    -------
    cutted_element_list : (list)
        List after cut.

    '''
    element_num = len(input_list)
    element_length = len(input_list[0])
    # get least element length, all element will cut into this length.
    for i in range(element_num):
        current_len = len(input_list[i])
        if element_length > current_len:
            element_length = current_len
    # Then generate cutted element lists
    cutted_element_list = []
    for i in range(element_num):
        if start == 'head':
            cutted_element_list.append(input_list[i][:element_length])
        elif start == 'tail':
            current_length = len(input_list[i])
            cutted_element_list.append(input_list[i][current_length-element_length:])
    return cutted_element_list

#%% Function 8: 2p runname style change.
def Change_Runid_Style(input_list):
    '''
    Change style of runid interpreter. change between 'Run001' into '1-001'.

    Parameters
    ----------
    input_list : (list)
        List of run ids.

    Returns
    -------
    another_style : (list)
        Another style of runid.

    '''
    testor = input_list[0]
    another_style = []
    if testor[:2] == '1-': # meaning we have '1-001' mode names.
        for i,c_run in enumerate(input_list):
            another_style.append('Run'+c_run[2:])
    elif testor[:3] == 'Run': # meaning we have 'Run001' mode names.
        for i,c_run in enumerate(input_list):
            another_style.append('1-'+c_run[3:])
    else:
        raise IOError('Invalid run style.')
    
    
    
    return another_style