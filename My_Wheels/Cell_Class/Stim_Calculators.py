
'''
This script include multiple vital functions for cell tuning calculation.
Codes following will generate:
1. Single cell tuning property(t values of OD/Orien/Color etc..)
2. T test maps (in format cell t value matrix)
3. Cell Response Curve. avr/std of each ID response.(for all cells)
4. This tool need to be updatable.
'''
#%%
from My_Wheels.Cell_Class.Format_Cell import Cell
import My_Wheels.Stim_Dic_Tools as SDT
import numpy as np
from Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
import Cell_Class.Cell_Class_Tools as Class_Tools
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Decorators import Timer
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import pandas as pd

class Stim_Cells(Cell):
    
    def __init__(self,*args, **kwargs): # Success all functions & variables in parent class Cell.
        super().__init__(*args, **kwargs)
        # And copy all variables.
        self.Calculate_All_CR_Response()
        self.Calculate_All_Stim_Response()

    def Get_CR_Response_Core(self,runname='1-006',head_extend=3,tail_extend = 3):
        # Input a frame, generate CR response of all cells in given frame.
        style2_runname = 'Run'+runname[2:]
        c_stim_frame_align = self.Stim_Frame_Align[style2_runname]
        self.Cell_Response = {}
        condition_frames = SDT.Condition_Response_Frames(c_stim_frame_align,head_extend,tail_extend)
        # get each condition have same length.
        condition_length = 65535
        all_conditions = list(condition_frames.keys())
        for i in range(len(all_conditions)):# get proper length. Use least length.
            current_cond_length = len(condition_frames[all_conditions[i]][0])
            if current_cond_length < condition_length:
                condition_length = current_cond_length
        for i in range(len(all_conditions)):# cut too long condition.
            current_condition = condition_frames[all_conditions[i]] # This is shallow copy so change below will be returned too.
            for j in range(len(current_condition)):
                current_condition[j] = current_condition[j][:condition_length]

        # Calculate Condition Response.
        c_Z_frame = self.Z_Frames[runname]
        all_cell_cr_dic = {}
        for j,cc in enumerate(self.acn):
            c_Z_train = np.array(c_Z_frame[cc])
            all_cell_cr_dic[cc] = {}
            for i,c_condition in enumerate(all_conditions):
                c_frame_lists = condition_frames[c_condition]
                Z_matrix = np.zeros(shape = (len(c_frame_lists[0]),len(c_frame_lists)),dtype = 'f8')
                for j in range(len(c_frame_lists)):
                    cs_cond = c_frame_lists[j]
                    Z_matrix[:,j] = c_Z_train[cs_cond]
                all_cell_cr_dic[cc][c_condition] = Z_matrix
        return all_cell_cr_dic
    
    def Calculate_All_CR_Response(self):
        if self.od_type != False:
            self.od_CR_Response = self.Get_CR_Response_Core(runname = self.odrun)
        if self.orien_type != False:
            self.orien_CR_Response = self.Get_CR_Response_Core(runname = self.orienrun)
        if self.color_type != False:
            self.color_CR_Response = self.Get_CR_Response_Core(runname = self.colorrun)
    
    def Calculate_All_Stim_Response(self):
        # Must be done after All CR_Response Calculation finish.
        self.Stim_Reponse_Dics = {}
        if self.od_type == 'OD_2P':
            self.Stim_Reponse_Dics['OD'] = Class_Tools.All_Cell_Condition_Combiner(self.od_CR_Response,Condition_Dics=Stim_ID_Combiner('OD_2P'))
        if self.orien_type == 'G16':# orientation will return both dir and orientation methods.
            self.Stim_Reponse_Dics['Oriens'] = Class_Tools.All_Cell_Condition_Combiner(self.orien_CR_Response,Condition_Dics=Stim_ID_Combiner('G16_Oriens'))
            self.Stim_Reponse_Dics['Dirs'] = Class_Tools.All_Cell_Condition_Combiner(self.orien_CR_Response,Condition_Dics=Stim_ID_Combiner('G16_Dirs'))
        if self.color_type == 'Hue7Orien4':
            self.Stim_Reponse_Dics['Colors'] = Class_Tools.All_Cell_Condition_Combiner(self.color_CR_Response,Condition_Dics=Stim_ID_Combiner('Hue7Orien4_Colors'))
            
    def Plot_Stim_Response(self,stim = 'OD',stim_on = (3,6),error_bar = True,figsize =(15,15)):
        # This function can plot stim response curve. Add multiple visualization.
        foldername = stim+'_Response_Curves'
        c_savepath = ot.join(self.wp,foldername)
        ot.mkdir(c_savepath)
        c_stimdata = self.Stim_Reponse_Dics[stim] # get all cell stim data.
        if stim == 'OD':
            graph_shape = (3,5)
        elif stim == 'Oriens':
            graph_shape = (3,3)
        elif stim == 'Dirs':
            graph_shape = (3,8)
        elif stim == 'Colors':
            graph_shape = (2,4)
        for i,cc in enumerate(self.acn):
            cc_stim_data = c_stimdata[cc]
            subgraph_num = len(cc_stim_data)
            all_subgraph_name = list(cc_stim_data.keys())
            y_max = 0# get y sticks
            y_min = 65535
            response_plot_dic = {}
            for j,c_subgraph in enumerate(all_subgraph_name):
                current_graph_response = cc_stim_data[c_subgraph]
                average_plot = current_graph_response.mean(1)
                se_2 = current_graph_response.std(1)/np.sqrt(current_graph_response.shape[1])*2
                response_plot_dic[c_subgraph] = (average_plot,se_2)
                # renew y min and y max.
                if average_plot.min() < y_min:
                    y_min = average_plot.min()
                if average_plot.max() > y_max:
                    y_max = average_plot.max()
            y_range = [y_min-0.3,y_max+0.3]
            # then we will plot..
            fig,ax = plt.subplots(graph_shape[0],graph_shape[1],figsize = figsize)
            fig.suptitle('Cell'+str(cc)+'_'+foldername, fontsize=30)
            for j in range(subgraph_num):
                current_col = j%graph_shape[1]
                current_row = j//graph_shape[1]
                current_graph_name = all_subgraph_name[j]
                current_data = response_plot_dic[current_graph_name]
                frame_num = len(current_data[0])
                # Start plot
                ax[current_row,current_col].hlines(y_range[0]+0.05, stim_on[0],stim_on[1],color="r")
                ax[current_row,current_col].set_ylim(y_range)
                ax[current_row,current_col].set_xticks(range(frame_num))
                ax[current_row,current_col].set_title(current_graph_name)
                # Whether we plot error bar on graph.
                if error_bar == True:
                    ax[current_row,current_col].errorbar(range(frame_num),current_data[0],current_data[1],fmt = 'bo-',ecolor='g')
                else:
                    ax[current_row,current_col].errorbar(range(frame_num),current_data[0],fmt = 'bo-')
            # Save ploted graph.
            fig.savefig(c_savepath+r'\\'+str(cc)+'_Response.png',dpi = 180)
            plt.clf()
            plt.close()
        
    @Timer
    def Plot_All_Tuning_Curve(self):
    # This is just a full version of all 4 stim response plot.
        print('This will cost some time, just be patient.')
        all_stim_types = list(self.Stim_Reponse_Dics.keys())
        for i,c_stim in enumerate(all_stim_types):
            self.Plot_Stim_Response(stim=c_stim)

    ############### Functions above are CR responses, Below are Tunings #############
    def T_Calculator_Core(self,cr_response,A_set,B_set,used_frame = [4,5]):
        # This function will generate single t test. Input all cell CR Resposne and Generated Subdic
        # Cycle all graphs.
        ttest_frame = pd.DataFrame(columns = self.acn,index = ['t_value','p_value','A_reponse','B_response'])
        # concat all response.
        for i,cc in enumerate(self.acn):
            cc_response = cr_response[cc]
            # get all data of A set and B set.
            for j,c_cond in enumerate(A_set):# get a set data.
                if j == 0: # for first condition
                    A_response = cc_response[c_cond]
                else:
                    A_response = np.hstack((A_response,cc_response[c_cond]))
            for j,c_cond in enumerate(B_set):# get a set data.
                if j == 0: # for first condition
                    B_response = cc_response[c_cond]
                else:
                    B_response = np.hstack((B_response,cc_response[c_cond]))
            used_A_response = A_response[used_frame,:].flatten()
            used_B_response = B_response[used_frame,:].flatten()
            c_t,c_p = ttest_ind(used_A_response,used_B_response)
            A_avr = used_A_response.mean()
            B_avr = used_B_response.mean()
            ttest_frame[cc] = [c_t,c_p,A_avr,B_avr]
        return ttest_frame
        
            
    def Calculate_All_T_Graphs(self):
        # return 3 dics, OD_t_graphs,Orien_t_graphs,Color_t_graphs
        self.OD_t_graphs = {}
        self.Orien_t_graphs = {}
        self.Color_t_graphs = {}
        if self.od_type == 'OD_2P':
            od_subdics = Sub_Dic_Generator('OD_2P')
            all_od_graphs = list(od_subdics.keys())
            # get each t graphs.
            for i,c_od_graph in enumerate(all_od_graphs):
                c_A,c_B = od_subdics[c_od_graph]
                c_od_graph_t = self.T_Calculator_Core(self.od_CR_Response,c_A,c_B)
                self.OD_t_graphs[c_od_graph] = c_od_graph_t
        if self.orien_type == 'G16':
            orien_subdics = Sub_Dic_Generator('G16_2P')
            all_orien_graphs = list(orien_subdics.keys())
            for i,c_orien_graph in enumerate(all_orien_graphs):
                c_A,c_B = orien_subdics[c_orien_graph]
                c_orien_graph_t = self.T_Calculator_Core(self.orien_CR_Response,c_A,c_B)
                self.Orien_t_graphs[c_orien_graph] = c_orien_graph_t
        if self.color_type == 'Hue7Orien4':
            color_subdics = Sub_Dic_Generator('HueNOrien4',para = 'Default')
            all_color_graphs = list(color_subdics.keys())
            for i,c_color_graph in enumerate(all_color_graphs):
                c_A,c_B = color_subdics[c_color_graph]
                c_color_graph_t = self.T_Calculator_Core(self.color_CR_Response,c_A,c_B)
                self.Color_t_graphs[c_color_graph] = c_color_graph_t

    def Plot_T_Graphs(self,thres = 0.05):
        # This will physically plot all t graphs. If you really need it.
        if not hasattr(self,'OD_t_graphs'):
            print('T calculation not finished. We need to calculate it first.')
            self.Calculate_All_T_Graphs()

        OD_path = ot.join(self.wp,'OD_T_Graphs')
        orien_path = ot.join(self.wp,'Orien_T_Graphs')
        color_path = ot.join(self.wp,'Color_T_Graphs')
        ot.mkdir(OD_path)
        ot.mkdir(orien_path)
        ot.mkdir(color_path)
        # OD graph
        OD_graph_name = list(self.OD_t_graphs)
        for i,c_od_graph in enumerate(OD_graph_name):
            c_t_series = self.OD_t_graphs[c_od_graph].loc['t_value',:]
            c_p_series = self.OD_t_graphs[c_od_graph].loc['p_value',:]
            t_series_thresed = c_t_series*(c_p_series<thres)
            visualized_t_graph = self.Generate_Weighted_Cell(t_series_thresed)
            fig = plt.figure(figsize = (15,15))
            plt.title(c_od_graph+' t Map',fontsize=36)
            fig = sns.heatmap(visualized_t_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
            fig.figure.savefig(OD_path+r'\\'+c_od_graph+'_t_Map.png')
            plt.close()
        # Orien graph
        Orien_graph_name = list(self.Orien_t_graphs)
        for i,c_orien_graph in enumerate(Orien_graph_name):
            c_t_series = self.Orien_t_graphs[c_orien_graph].loc['t_value',:]
            c_p_series = self.Orien_t_graphs[c_orien_graph].loc['p_value',:]
            t_series_thresed = c_t_series*(c_p_series<thres)
            visualized_t_graph = self.Generate_Weighted_Cell(t_series_thresed)
            fig = plt.figure(figsize = (15,15))
            plt.title(c_orien_graph+' t Map',fontsize=36)
            fig = sns.heatmap(visualized_t_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
            fig.figure.savefig(orien_path+r'\\'+c_orien_graph+'_t_Map.png')
            plt.close()
        # color graph
        Color_graph_name = list(self.Color_t_graphs)
        for i,c_color_graph in enumerate(Color_graph_name):
            c_t_series = self.Color_t_graphs[c_color_graph].loc['t_value',:]
            c_p_series = self.Color_t_graphs[c_color_graph].loc['p_value',:]
            t_series_thresed = c_t_series*(c_p_series<thres)
            visualized_t_graph = self.Generate_Weighted_Cell(t_series_thresed)
            fig = plt.figure(figsize = (15,15))
            plt.title(c_color_graph+' t Map',fontsize=36)
            fig = sns.heatmap(visualized_t_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
            fig.figure.savefig(color_path+r'\\'+c_color_graph+'_t_Map.png')
            plt.close()
            
        
    def Calculate_Cell_Tunings(self,thres=0.05):
        # Calculate all cell tunings, return a cell data frame of tuning t/p.
        OD_stims = ['L-0','R-0','OD']
        Orien_stims = ['Orien0-0','Orien22.5-0','Orien45-0','Orien67.5-0','Orien90-0','Orien112.5-0','Orien135-0','Orien157.5-0']
        Color_stims = ['Red-White','Yellow-White','Green-White','Cyan-White','Blue-White','Purple-White']
        Tunings = ['Best_Eye','Best_Orien','Best_Color','OD_index','Orien_index']
        all_stim = OD_stims+Orien_stims+Color_stims+Tunings
        self.all_cell_tunings = pd.DataFrame(columns = self.acn,index=all_stim)
        self.all_cell_tunings_p_value = pd.DataFrame(columns = self.acn,index=all_stim[:-5])# record significant status.
        # Get 3 types of Tunings
        ODs = self.OD_t_graphs
        for i,c_od in enumerate(OD_stims):
            c_t_graph = ODs[c_od].loc['t_value',:]
            self.all_cell_tunings.loc[c_od,:] = c_t_graph
            c_p_graph = ODs[c_od].loc['p_value',:]
            self.all_cell_tunings_p_value.loc[c_od,:] = c_p_graph
        Oriens = self.Orien_t_graphs
        for i,c_orien in enumerate(Orien_stims):
            c_t_graph = Oriens[c_orien].loc['t_value',:]
            self.all_cell_tunings.loc[c_orien,:] = c_t_graph
            c_p_graph = Oriens[c_orien].loc['p_value',:]
            self.all_cell_tunings_p_value.loc[c_orien,:] = c_p_graph
        Colors = self.Color_t_graphs
        for i,c_color in enumerate(Color_stims):
            c_t_graph = Colors[c_color].loc['t_value',:]
            self.all_cell_tunings.loc[c_color,:] = c_t_graph
            c_p_graph = Colors[c_color].loc['p_value',:]
            self.all_cell_tunings_p_value.loc[c_color,:] = c_p_graph
        # get max eye,orien,color. This will need a cycle on cells.
        for i,cc in enumerate(self.acn):
            c_tunings = self.all_cell_tunings[cc]
            c_tunings_p = self.all_cell_tunings_p_value[cc]
            # get best eye and OD_index
            if c_tunings['OD']>0 and c_tunings_p['OD']<thres:
                c_best_eye = 'LE'
            elif c_tunings['OD']<0 and c_tunings_p['OD']<thres:
                c_best_eye = 'RE'
            else:
                c_best_eye = 'False'
            c_od_index = (c_tunings['L-0']-c_tunings['R-0'])/(c_tunings['L-0']+c_tunings['R-0'])
            # Then best orientation
            all_oriens = c_tunings[3:11]
            best_orien = all_oriens[all_oriens == all_oriens.max()].index[0]
            if c_tunings_p[best_orien]<thres:
                c_best_orien = best_orien[:-2]
            else:
                c_best_orien = 'False'
            best_orien_numeric = float(best_orien[5:-2])
            if best_orien_numeric < 90:# +90 to get counter orien.
                counter_orien_num = best_orien_numeric+90
            else:
                counter_orien_num = best_orien_numeric-90
            if counter_orien_num%45 == 0: # No need for .5
                counter_orien = 'Orien'+str(int(counter_orien_num))+'-0'
            else: # .5 need to keep.
                counter_orien = 'Orien'+str((counter_orien_num))+'-0'
            best_orien_response = c_tunings[best_orien]
            counter_orien_response = c_tunings[counter_orien]
            c_orien_index = (best_orien_response-counter_orien_response)/(best_orien_response+counter_orien_response)
            # last, get colors.
            all_colors = c_tunings[11:17]
            best_color = all_colors[all_colors == all_colors.max()].index[0]
            if c_tunings_p[best_color]<thres:
                c_best_color = best_color[:-6]
            else:
                c_best_color = 'False'
            self.all_cell_tunings[cc]['Best_Eye'] = c_best_eye
            self.all_cell_tunings[cc]['Best_Orien'] = c_best_orien
            self.all_cell_tunings[cc]['Best_Color'] = c_best_color
            self.all_cell_tunings[cc]['OD_index'] = c_od_index
            self.all_cell_tunings[cc]['Orien_index'] = c_orien_index
            
    def Get_Labels(self,runname,label_type = 'OD+Orien',have_isi = True):
        # get label of given frame, can be OD,OD-orien,orien,dir,color etc.
        pass
    
    def Calculate_All(self):
        # This will calculate all nacessary result and save the class in folder.
        pass
    
    
#%%
if __name__ == '__main__':
    day_folder = r'F:\_Data_Temp\220630_L76_2P'
    test_cell = Stim_Cells(day_folder,od = 6,orien = 7, color = 8)
    # test_cell.Get_All_Stim_Response()
    # test_cell.Plot_Stim_Response(stim = 'OD')
    test_cell.Calculate_All_T_Graphs()
    test_cell.Calculate_Cell_Tunings()
    