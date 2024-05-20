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
# from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
import random

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



class Classify_Analyzer(object):
    
    name = r'UMAP Recover Stim map processing tools'

    def __init__(self,ac,model,spon_frame,od = True,orien = True,color = True,isi = True) -> None:
        self.ac = ac
        self.model = model
        self.spon_frame = spon_frame
        # some basic cut and embeddings
        self.stim_frame,self.stim_label = ac.Combine_Frame_Labels(od = od,orien = orien,color = color,isi = isi)
        self.stim_embeddings = self.model.transform(self.stim_frame)
        self.spon_embeddings = self.model.transform(self.spon_frame)


    def Train_SVM_Classifier(self,predict = True,C = 1):

        self.svm_classifier,self.svm_fitscore = SVM_Classifier(embeddings=self.stim_embeddings,label = self.stim_label,C = C)
        if predict == True:
            self.spon_label = SVC_Fit(self.svm_classifier,data = self.spon_embeddings,thres_prob = 0)
        else:
            print('No Prediction, be cautious.')
    
    def Get_Func_Maps(self,method = 'Stim',od = True,orien = True,color = True): # method can be Stim,Spon.
        # test whether we have spon label here.
        if method == 'Spon' or method == 'Compare':
            try:
                self.spon_label
            except AttributeError:
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

    def Get_Stim_Spon_Compare(self,od = True,orien = True,color = True): # this method will return a compare graph of stim and recovered.
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
            
                   
    def Similarity_Compare_All(self,id_lists = [1,3,5,7]): 
        real_corr = []
        random_corr = []
        for i,c_id in enumerate(id_lists):
            cid_stim_frame,_ = Select_Frame(self.stim_frame,self.stim_label,used_id=[c_id])
            cid_spon_frame,_ = Select_Frame(self.spon_frame,self.spon_label,used_id=[c_id])
            if len(cid_spon_frame) == 0: # no match on spon situation.
                continue
            c_pattern = cid_stim_frame.mean(0)
            for j,c_frame in enumerate(cid_spon_frame):
                c_r,_ = stats.pearsonr(c_pattern,c_frame)
                real_corr.append(c_r)
                # rand select a single spon frame.
                rand_frameid = random.randint(0,len(self.spon_frame)-1) # avoid out of boundary.
                rand_r,_ = stats.pearsonr(c_pattern,np.array(self.spon_frame)[rand_frameid,:])
                random_corr.append(rand_r)
        return real_corr,random_corr


    def Average_Corr_Core(self,id_list = [1,3,5,7]):# generate single map avr and random. Will return real and random average corr.
        used_spon,_ = Select_Frame(self.spon_frame,self.spon_label,id_list)
        spon_num = len(used_spon)
        if spon_num>0:
            random_indices = np.random.choice(np.array(self.spon_frame).shape[0], size=spon_num, replace=False)
            template,_ = Select_Frame(self.stim_frame,self.stim_label,id_list)
            template_avr = template.mean(0)
            used_spon_avr = used_spon.mean(0)
            random_selected = np.array(self.spon_frame)[random_indices].mean(0)
            real_corr,_ = stats.pearsonr(template_avr,used_spon_avr)
            rand_corr,_ = stats.pearsonr(template_avr,random_selected)
            return real_corr,rand_corr,spon_num
        else:
            return 0,0,spon_num



    def Similarity_Compare_Average(self,od = True,orien = True,color = True):# average graph.
        try: # if recover map not generate, generate here.
            self.compare_recover
        except AttributeError:
            self.Get_Stim_Spon_Compare()
        self.Avr_Similarity = pd.DataFrame(columns=['PearsonR','Network','Data','MapType'])
        if od == True:
            LE_corr,LE_corr_shuffle,LE_corr_Num = self.Average_Corr_Core([1,3,5,7])
            if LE_corr_Num != 0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [LE_corr,'LE','Real Data','OD']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [LE_corr_shuffle,'LE','Shuffle','OD']
            RE_corr,RE_corr_shuffle,RE_corr_Num = self.Average_Corr_Core([2,4,6,8])
            if RE_corr_Num != 0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [RE_corr,'RE','Real Data','OD']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [RE_corr_shuffle,'RE','Shuffle','OD']
        if orien == True:
            Orien0_corr,Orien0_corr_shuffle,Orien0_corr_num = self.Average_Corr_Core([9])
            if Orien0_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien0_corr,'Orien0','Real Data','Orien']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien0_corr_shuffle,'Orien0','Shuffle','Orien']
            Orien45_corr,Orien45_corr_shuffle,Orien45_corr_num = self.Average_Corr_Core([11])
            if Orien45_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien45_corr,'Orien45','Real Data','Orien']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien45_corr_shuffle,'Orien45','Shuffle','Orien']
            Orien90_corr,Orien90_corr_shuffle,Orien90_corr_num = self.Average_Corr_Core([13])
            if Orien90_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien90_corr,'Orien90','Real Data','Orien']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien90_corr_shuffle,'Orien90','Shuffle','Orien']
            Orien135_corr,Orien135_corr_shuffle,Orien135_corr_num = self.Average_Corr_Core([15])
            if Orien135_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien135_corr,'Orien135','Real Data','Orien']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Orien135_corr_shuffle,'Orien135','Shuffle','Orien']
        if color == True:
            Red_corr,Red_corr_shuffle,Red_corr_num = self.Average_Corr_Core([17])
            if Red_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Red_corr,'Red','Real Data','Color']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Red_corr_shuffle,'Red','Shuffle','Color']
            Green_corr,Green_corr_shuffle,Green_corr_num = self.Average_Corr_Core([19])
            if Green_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Green_corr,'Green','Real Data','Color']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Green_corr_shuffle,'Green','Shuffle','Color']
            Blue_corr,Blue_corr_shuffle,Blue_corr_num = self.Average_Corr_Core([21])
            if Blue_corr_num !=0:
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Blue_corr,'Blue','Real Data','Color']
                self.Avr_Similarity.loc[len(self.Avr_Similarity)] = [Blue_corr_shuffle,'Blue','Shuffle','Color']



if __name__ == '__main__':
    pass