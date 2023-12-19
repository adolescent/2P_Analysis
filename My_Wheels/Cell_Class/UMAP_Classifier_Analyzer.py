'''

This class will do stats and recovery of most data using UMAP method, this will generate most graphs we need here.


'''

class UMAP_Analyzer(object):
    
    name = r'UMAP Recover Stim map processing tools'

    def __init__(self,ac,umap_model,spon_frame) -> None:
        pass

    def Train_SVM_Classifier(self,method = 'Stim',od = True,orien = True,color = True):
        pass
    def SVM_Predict_Spon_ID_Seires(self,svm_model,spon_series):# shuffle can use this api too.
        pass
    def Get_Single_ID_Frames_Core(self,input_id_list,frame_list):# both stim and spon can use this function.
        pass # return avr graph and selected graph lists.
    def Similarity_Compare_All(self,pattern,all_frame,label_id,wanted_label): # this will return all similarity vs pattern of all id given. random select given here too.
        pass
    def Similarity_Compare_Combine(self,pattern,all_frame,label_id,wanted_label): # this will return avr similarity, only one value will be given.
        pass


if __name__ == '__main__':
    pass