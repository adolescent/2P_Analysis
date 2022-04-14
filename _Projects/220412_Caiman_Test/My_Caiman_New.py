# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:18:27 2022

@author: ZR

This part is used for caiman understanding.

After using this, I shall be able to use 

"""


import Graph_Operation_Kit as gt
import bokeh.plotting as bpl
from tqdm import tqdm
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
