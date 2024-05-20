'''
This script will calculate time course information of network repeatance.
'''


from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
# from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
# import umap
# import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
import random


class Series_TC_info(object):
    name = r'Analyze timeourse information'
    def __init__(self,input_series,od = True,orien = True,color = True,fps = 1.301) -> None:
        self.input_series = input_series
        self.od_flag = od
        self.orien_flag = orien
        self.color_flag = color
        self.fps = fps
        self.series_len = len(self.input_series)

    def Freq_Estimation(self,type = 'Event'): # Event or Frame
        if self.od_flag == False:
            od_freq = 0
        else:
            od_train = (self.input_series>0)*(self.input_series<9)
            self.od_event,self.od_event_len = Label_Event_Cutter(od_train)
            if type == 'Event':
                od_freq = len(self.od_event)*self.fps/self.series_len
            elif type == 'Frame':
                od_freq = (od_train.sum())*self.fps/self.series_len

        if self.orien_flag == False:
            orien_freq = 0
        else:
            orien_train = (self.input_series>8)*(self.input_series<17)
            self.orien_event,self.orien_event_len = Label_Event_Cutter(orien_train)
            if type == 'Event':
                orien_freq = len(self.orien_event)*self.fps/self.series_len
            elif type == 'Frame':
                orien_freq = (orien_train.sum())*self.fps/self.series_len

        if self.color_flag == False:
            color_freq = 0
        else:
            color_train = (self.input_series>16)
            self.color_event,self.color_event_len = Label_Event_Cutter(color_train)
            if type == 'Event':
                color_freq = len(self.color_event)*self.fps/self.series_len
            elif type == 'Frame':
                color_freq = (color_train.sum())*self.fps/self.series_len

        return od_freq,orien_freq,color_freq

