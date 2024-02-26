'''
Use 220902-L76-18M as example, show cell response and t maps.

'''


#%%

from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Plotter.Line_Plotter import EZLine
from tqdm import tqdm
import cv2

work_path = r'D:\_Path_For_Figs\Fig1_Data_Description'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(r'D:\_All_Spon_Datas_V1\L76_18M_220902\Global_Average_cai.tif',0) # read as 8 bit gray scale map.

#%% Regenerate Cell tuning response curve. To get good response plot.
# ac.Calculate_All_CR_Response()
# ac.Calculate_All_Stim_Response()
# ac.Save_Class(expt_folder,'Cell_Class')
print(ac.Stim_Reponse_Dics['Oriens'][23]['Orien0.0'].shape)
print(ac.Stim_Reponse_Dics['OD'][23]['L_All'].shape)
print(ac.Stim_Reponse_Dics['Colors'][23]['Red'].shape)
ac.Plot_Stim_Response(stim = 'Colors')
ac.Plot_Stim_Response(stim = 'OD')
ac.Plot_Stim_Response(stim = 'Oriens')
#%% Select a Cell of Color, OD and Orien.
# Cell 274, as a good RE-Red-Orien 135 cell.
# Cell 13 as a good LE-Red-Orien  cell
# Cell 293 as a good RE-Blue-Orien  cell.
# generate a pd dataframe, using frame, stim type as group, aiming to plot easily with seaborn.
# cell_example_list = [13,274,293] # not very good shape.
cell_example_list = [47,322,338]
all_response_frame = pd.DataFrame(columns=['Cell','Frame','Stim_Type','Response'])
def Pandas_Filler(data_frame,c_response_frame,cell_name,stim_name):
    frame_num,condition_num = c_response_frame.shape
    for i in range(frame_num):
        for j in range(condition_num):
            data_frame.loc[len(data_frame)] = [cell_name,i,stim_name,c_response_frame[i,j]]
    return data_frame

for i,cc in enumerate(cell_example_list):
    c_LE_response = ac.Stim_Reponse_Dics['OD'][cc]['L_All']
    c_RE_response = ac.Stim_Reponse_Dics['OD'][cc]['R_All']
    c_H_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien0.0']
    c_A_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien45.0']
    c_V_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien90.0']
    c_O_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien135.0']
    c_Red_response = ac.Stim_Reponse_Dics['Colors'][cc]['Red']
    c_Green_response = ac.Stim_Reponse_Dics['Colors'][cc]['Green']
    c_Blue_response = ac.Stim_Reponse_Dics['Colors'][cc]['Blue']
    all_response_frame = Pandas_Filler(all_response_frame,c_LE_response,cc,'LE')
    all_response_frame = Pandas_Filler(all_response_frame,c_RE_response,cc,'RE')
    all_response_frame = Pandas_Filler(all_response_frame,c_H_response,cc,'Orien0')
    all_response_frame = Pandas_Filler(all_response_frame,c_A_response,cc,'Orien45')
    all_response_frame = Pandas_Filler(all_response_frame,c_V_response,cc,'Orien90')
    all_response_frame = Pandas_Filler(all_response_frame,c_O_response,cc,'Orien135')
    all_response_frame = Pandas_Filler(all_response_frame,c_Red_response,cc,'Red')
    all_response_frame = Pandas_Filler(all_response_frame,c_Green_response,cc,'Green')
    all_response_frame = Pandas_Filler(all_response_frame,c_Blue_response,cc,'Blue')

#%% Plot maps using subgraphs.
plt.cla()
plt.clf()
# cell_example_list = [13,274,293] # already defined
frame_example_list = ['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue']

fig,ax = plt.subplots(len(cell_example_list),len(frame_example_list),figsize = (13,4),dpi = 180)
fig.tight_layout(h_pad=0.5)
# for axs in ax.flat:
#     axs.set(ylabel='Z Score')
#     axs.label_outer()
# plotter of all graph.
for i,cc in enumerate(cell_example_list):
    cc_group = all_response_frame.groupby('Cell').get_group(cc)
    for j,c_condition in enumerate(frame_example_list):
        c_graph = cc_group.groupby('Stim_Type').get_group(c_condition)
        c_graph = c_graph[c_graph['Frame']<10]
        ax[i,j].set(ylim = (-1.2,4))
        ax[i,j].set_xticks([2,4,6,8])
        sns.lineplot(data = c_graph,x = 'Frame',y = 'Response',ax = ax[i,j])
        ax[i,j].axvspan(xmin = 3,xmax = 6,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
        if i == 2:
            # ax[i,j].hlines(y = -1,xmin = 3,xmax = 6,linewidth=2, color='r')
            ax[i,j].set_xlabel(c_condition)
        if j == 0: # for the first row
            ax[i,j].set_ylabel(f'Z Score')
            ax[i,j].xaxis.set_visible(False) # off x axis
            ax[i,j].spines['top'].set_visible(False)  # off frames
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(False)
            ax[i,j].spines['left'].set_visible(True)
            if i == 2:
                ax[i,j].xaxis.set_visible(True)
                ax[i,j].spines['bottom'].set_visible(True)
        elif i ==2:
            ax[i,j].xaxis.set_visible(True)
            ax[i,j].yaxis.set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(True)
        else:
            ax[i,j].axis('off')
# add subtitles
for i, title in enumerate(cell_example_list):
    fig.text(0.5, 1-(i*.95)/3,f'Cell {title}', va='center', ha='center', fontsize=12)
        # ax[i,j].set_title(f'Response {c_condition}')
        
#%% Plot 3 Cell with red circle without show.
annotated_graph = ac.Annotate_Cell([47,322,338])
image=cv2.cvtColor(annotated_graph,cv2.COLOR_BGR2RGB)
cv2.imwrite(work_path+'\Annotated_Graph.png',image)