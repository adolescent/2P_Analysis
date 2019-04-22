# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:29:22 2019

@author: adolescent
"""
import numpy as np
import cv2
import scipy.ndimage
import functions_OD as pp
import dill
import skimage.morphology
import skimage.measure

dill.load_session('Step1_Variable.pkl')#载入前一个任务的变量，方便变量继承
class Cell_Found():#定义类
    name =r'Cell_Found'#定义class的属性，如果没有__init__的内容就会以这里的作为类属性。
    def __init__(self,show_gain,save_folder,thres):
        