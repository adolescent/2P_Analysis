# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:10:38 2019

@author: ZR

This Module is Used to do List File Operations, making such work easier.

"""

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
        return out_str
    else: # If want same tail
        out_str = []
        for i in range(len(A)):
            out_str.append(str(A[i])+dilimit+str(B[0]))
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
#%% Function3: List To Dictionary
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
