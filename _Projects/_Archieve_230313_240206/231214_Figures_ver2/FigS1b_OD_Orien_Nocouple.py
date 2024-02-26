'''
This graph will show no coupling of OD and Orientation preference, Plot the polar map as we use before.

As a stat result, all point's data will be put on the graph.

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

all_point_path = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_point_path.pop(4)
all_point_path.pop(6)
# get Locations
all_cell_info = pd.DataFrame(columns=['Best_Orien','OD_t','Location','Cell_name']) 
counter = 0
for i,c_loc in tqdm(enumerate(all_point_path)):
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    for j,cc in enumerate(c_ac.acn):
        if c_ac.all_cell_tunings[cc]['Best_Orien'] != 'False':
            c_orien_tunings = float(c_ac.all_cell_tunings[cc]['Best_Orien'][5:])
            c_od_t = c_ac.all_cell_tunings[cc]['OD']
            all_cell_info.loc[counter] = [c_orien_tunings,c_od_t,c_loc.split('\\')[-1],cc]
            counter +=1
#%%############################# FIG S1a - POLAR MAPS ################################
# first, Plot each data point in different clor.
plt.cla()
plt.clf()
fig = plt.figure()
ax = plt.axes(polar=True)
ax.set_rlim(30,-20)
all_locs_group = all_cell_info.groupby('Location')
for i,c_loc in enumerate(all_locs_group):
    group = c_loc[1].groupby('Best_Orien')['OD_t']
    r = np.array(group.mean(1)).flatten()
    theta = 4*np.pi/360 *(np.arange(0, 180, 22.5))
    # se = np.array(group.std()).flatten()
    se = np.array(group.sem(1)).flatten()
    # Duplicate the first data point at the end
    theta = np.append(theta, theta[0])
    r = np.append(r, r[0])
    se = np.append(se,se[0])
    # ax.plot(theta, r)
    # ax.errorbar(theta,r,yerr=se)
    ax.plot(theta,r,alpha = 0.3)
    ax.fill_between(theta,r-se, r+se, alpha=0.1)


# Second, we plot global average graph below.
global_group = all_cell_info.groupby('Best_Orien')['OD_t']
r = np.array(global_group.mean(1)).flatten()
theta = 4*np.pi/360 *(np.arange(0, 180, 22.5))
se = np.array(global_group.sem(1)).flatten()
# Duplicate the first data point at the end
theta = np.append(theta, theta[0])
r = np.append(r, r[0])
se = np.append(se,se[0])
ax.plot(theta,r,alpha = 1,color = 'black',linewidth = 3)
ax.fill_between(theta,r-se, r+se, alpha=0.5,color = 'black')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xlabel('OD Preference In Different Orientation')
# ax.set_xticks(np.arange(0,180,22.5))
custom_labels = np.arange(0,180,22.5)  # Adjust the labels as needed
ax.set_xticklabels(custom_labels)
plt.show()
