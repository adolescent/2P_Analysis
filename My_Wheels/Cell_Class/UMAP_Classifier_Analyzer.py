'''

This class will do stats and recovery of most data using UMAP method, this will generate most graphs we need here.


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
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *

def Select_Frame(frame,label,used_id = [1,3,5,7]):
    frame = np.array(frame)# avoid pd frame bugs,
    used_loc = []
    selected_id_list = []
    for i,c_id in enumerate(label):
        if c_id in used_id:
            used_loc.append(i)
            selected_id_list.append(c_id)
    selected_frame = np.zeros(shape = (len(used_loc),frame.shape[1]),dtype='f8')
    for i,c_loc in enumerate(used_loc):
        selected_frame[i,:] = frame[c_loc,:]
    return selected_frame,selected_id_list



class UMAP_Analyzer(object):
    
    name = r'UMAP Recover Stim map processing tools'

    def __init__(self,ac,umap_model,spon_frame,od = True,orien = True,color = True,isi = True) -> None:
        self.ac = ac
        self.umap_model = umap_model
        self.spon_frame = spon_frame
        # some basic cut and embeddings
        self.stim_frame,self.stim_label = ac.Combine_Frame_Labels(od = od,orien = orien,color = color,isi = isi)
        self.stim_embeddings = self.umap_model.transform(self.stim_frame)
        self.spon_embeddings = self.umap_model.transform(self.spon_frame)


    def Train_SVM_Classifier(self,predict = True):

        self.svm_classifier,self.svm_fitscore = SVM_Classifier(embeddings=self.stim_embeddings,label = self.stim_label)
        if predict == True:
            self.spon_label = SVC_Fit(self.svm_classifier,data = self.spon_embeddings,thres_prob = 0)
        else:
            print('No Prediction, be cautious.')
    
    def Get_Func_Maps(self,method = 'Stim',od = True,orien = True,color = True): # method can be Stim,Spon.
        # test whether we have spon label here.
        if method == 'Spon' or method == 'Compare':
            try:
                self.spon_label
            except NameError:
                print('SVM prediction seems not be done.')
                self.Train_SVM_Classifier()
        # find data for process, and save structure.
        func_map = {}
        if method == 'Spon':
            all_series = self.spon_frame
            all_label = self.spon_label
        elif method == 'Stim':
            all_series = self.stim_frame
            all_label = self.stim_label
        # and generate all graphs here.
        if od == True:
            LE_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[1,3,5,7])
            LE_response_avr = LE_frame.mean(0)
            LE_response_map = self.ac.Generate_Weighted_Cell(LE_response_avr)
            RE_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[2,4,6,8])
            RE_response_avr = RE_frame.mean(0)
            RE_response_map = self.ac.Generate_Weighted_Cell(RE_response_avr)
            func_map['LE'] = (LE_response_avr,LE_response_map)
            func_map['RE'] = (RE_response_avr,RE_response_map)
        if orien == True:
            Orien0_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[9])
            Orien0_response_avr = Orien0_frame.mean(0)
            Orien0_map = self.ac.Generate_Weighted_Cell(Orien0_response_avr)
            Orien45_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[11])
            Orien45_response_avr = Orien45_frame.mean(0)
            Orien45_map = self.ac.Generate_Weighted_Cell(Orien45_response_avr)
            Orien90_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[13])
            Orien90_response_avr = Orien90_frame.mean(0)
            Orien90_map = self.ac.Generate_Weighted_Cell(Orien90_response_avr)
            Orien135_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[15])
            Orien135_response_avr = Orien135_frame.mean(0)
            Orien135_map = self.ac.Generate_Weighted_Cell(Orien135_response_avr)
            func_map['Orien0'] = (Orien0_response_avr,Orien0_map)
            func_map['Orien45'] = (Orien45_response_avr,Orien45_map)
            func_map['Orien90'] = (Orien90_response_avr,Orien90_map)
            func_map['Orien135'] = (Orien135_response_avr,Orien135_map)
        if color == True:
            Red_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[17])
            Red_response_avr = Red_frame.mean(0)
            Red_map = self.ac.Generate_Weighted_Cell(Red_response_avr)
            Green_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[19])
            Green_response_avr = Green_frame.mean(0)
            Green_map = self.ac.Generate_Weighted_Cell(Green_response_avr)
            Blue_frame,_ = Select_Frame(frame = all_series,label = all_label,used_id=[21])
            Blue_response_avr = Blue_frame.mean(0)
            Blue_map = self.ac.Generate_Weighted_Cell(Blue_response_avr)
            func_map['Red'] = (Red_response_avr,Red_map)
            func_map['Green'] = (Green_response_avr,Green_map)
            func_map['Blue'] = (Blue_response_avr,Blue_map)
            
        return func_map

    def Get_Stim_Spon_Compare(self,od = True,orien = True,color = True):
        self.stim_recover = self.Get_Func_Maps(method = 'Stim',od = od,orien = orien,color = color)
        self.spon_recover = self.Get_Func_Maps(method = 'Spon',od = od,orien = orien,color = color)
        self.compare_recover = {}
        if od == True:
            LE_compare = np.hstack((self.stim_recover['LE'][1],self.spon_recover['LE'][1]))
            RE_compare = np.hstack((self.stim_recover['RE'][1],self.spon_recover['RE'][1]))
            LE_compare[:,510:514] = 10
            RE_compare[:,510:514] = 10
            self.compare_recover['LE'] = LE_compare
            self.compare_recover['RE'] = RE_compare
            
        if orien == True:
            Orien0_compare = np.hstack((self.stim_recover['Orien0'][1],self.spon_recover['Orien0'][1]))
            Orien45_compare = np.hstack((self.stim_recover['Orien45'][1],self.spon_recover['Orien45'][1]))
            Orien90_compare = np.hstack((self.stim_recover['Orien90'][1],self.spon_recover['Orien90'][1]))
            Orien135_compare = np.hstack((self.stim_recover['Orien135'][1],self.spon_recover['Orien135'][1]))
            Orien0_compare[:,510:514] = 10
            Orien45_compare[:,510:514] = 10
            Orien90_compare[:,510:514] = 10
            Orien135_compare[:,510:514] = 10
            self.compare_recover['Orien0'] = Orien0_compare
            self.compare_recover['Orien45'] = Orien45_compare
            self.compare_recover['Orien90'] = Orien90_compare
            self.compare_recover['Orien135'] = Orien135_compare

        if color == True:
            Red_compare = np.hstack((self.stim_recover['Red'][1],self.spon_recover['Red'][1]))
            Green_compare = np.hstack((self.stim_recover['Green'][1],self.spon_recover['Green'][1]))
            Blue_compare = np.hstack((self.stim_recover['Blue'][1],self.spon_recover['Blue'][1]))
            Red_compare[:,510:514] = 10
            Green_compare[:,510:514] = 10
            Blue_compare[:,510:514] = 10
            self.compare_recover['Red'] = Red_compare
            self.compare_recover['Green'] = Green_compare
            self.compare_recover['Blue'] = Blue_compare
            
            
            
            
    
            


    def Similarity_Compare_All(self,pattern,all_frame,label_id,wanted_label): # this will return all similarity vs pattern of all id given. random select given here too.
        pass
    def Similarity_Compare_Combine(self,pattern,all_frame,label_id,wanted_label): # this will return avr similarity, only one value will be given.
        pass


if __name__ == '__main__':
    pass