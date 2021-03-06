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
    subtracted_list = list_A
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