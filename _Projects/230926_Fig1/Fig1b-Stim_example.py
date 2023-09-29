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

work_path = r'D:\_Path_For_Figs\Fig0_Timepoints'
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
