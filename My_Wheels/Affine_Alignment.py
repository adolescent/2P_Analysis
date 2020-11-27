# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:25:06 2020

@author: ZR
Alignment based on affine transformation.
This is used to align graph with bigger tremble, theoretically can fix stretch and rotation of images

"""

def Affine_Core(
        base,
        target,
        max_point = 100000,
        good_match = 0.15
        ):
    '''
    Use ORB method to do affine correlation.

    Parameters
    ----------
    base : (2D Array)
        Base graph.
    target : TYPE
        DESCRIPTION.
    max_point : TYPE, optional
        DESCRIPTION. The default is 100000.
    good_match : TYPE, optional
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    matched_graph : TYPE
        DESCRIPTION.

    '''
    return matched_graph