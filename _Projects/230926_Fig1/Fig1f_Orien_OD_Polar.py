'''
This script will generate polar map of Orientation and OD, showing the correlation relationship of these 2 variables.

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
#%% We get all cell OD t value and prefered orientation. no orien preference cell are ignored.
tuning_index = pd.DataFrame(-1.0,index = ac.acn,columns=['Best_Orien','OD_t'])
for i,cc in enumerate(ac.acn):
    if ac.all_cell_tunings[cc]['Best_Orien'] != 'False':
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        tuning_index.loc[cc]['Best_Orien'] = orien_tunings
        tuning_index.loc[cc]['OD_t'] = ac.all_cell_tunings[cc]['OD']
tuning_index = tuning_index.loc[tuning_index['Best_Orien']!= -1]
# group best orien and get t value's mean/std of oriens.
group = tuning_index.groupby('Best_Orien')
#%%
plt.cla()
plt.clf()
fig = plt.figure()
ax = plt.axes(polar=True)
r = np.array(group.mean()).flatten()
theta = 4*np.pi/360 *(np.arange(0, 180, 22.5))
# se = np.array(group.std()).flatten()
se = np.array(group.sem()).flatten()
# Duplicate the first data point at the end
theta = np.append(theta, theta[0])
r = np.append(r, r[0])
se = np.append(se,se[0])
# ax.plot(theta, r)
ax.set_rlim(10,-15)
# ax.errorbar(theta,r,yerr=se)
ax.plot(theta,r)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.fill_between(theta,r-se, r+se, alpha=0.2)
ax.set_xlabel('OD Preference In Different Orientation')
# ax.set_xticks(np.arange(0,180,22.5))
custom_labels = np.arange(0,180,22.5)  # Adjust the labels as needed
ax.set_xticklabels(custom_labels)
plt.show()
#%%#################################################################
# graph above is single location version, we make stats version below.
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
ot.Save_Variable(work_path,'All_Cell_OD_Orien',all_cell_info)
#%% Plot 2 graphs of OD-Orien.
# first, try plot different point seperately.
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
    ax.plot(theta,r,alpha = 0.5)
    ax.fill_between(theta,r-se, r+se, alpha=0.1)

# Plot global avr below!
global_group = all_cell_info.groupby('Best_Orien')['OD_t']
r = np.array(global_group.mean(1)).flatten()
theta = 4*np.pi/360 *(np.arange(0, 180, 22.5))
se = np.array(global_group.sem(1)).flatten()
# Duplicate the first data point at the end
theta = np.append(theta, theta[0])
r = np.append(r, r[0])
se = np.append(se,se[0])
# ax.plot(theta, r)
# ax.errorbar(theta,r,yerr=se)
ax.plot(theta,r,alpha = 1,color = 'black',linewidth = 3)
ax.fill_between(theta,r-se, r+se, alpha=0.5,color = 'black')

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xlabel('OD Preference In Different Orientation')
# ax.set_xticks(np.arange(0,180,22.5))
custom_labels = np.arange(0,180,22.5)  # Adjust the labels as needed
ax.set_xticklabels(custom_labels)
plt.show()
#%%##################################################################
#Plot 3, another approach. Here we stats all points across data points.
group = all_cell_info.groupby('Best_Orien')['OD_t']
plt.cla()
plt.clf()
fig = plt.figure()
ax = plt.axes(polar=True)
ax.set_rlim(20,-15)
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
ax.plot(theta,r,alpha = 0.7)
ax.fill_between(theta,r-se, r+se, alpha=0.2)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xlabel('OD Preference In Different Orientation')
# ax.set_xticks(np.arange(0,180,22.5))
custom_labels = np.arange(0,180,22.5)  # Adjust the labels as needed
ax.set_xticklabels(custom_labels)
plt.show()
#%%####################################################################################
# This is not a graph, but we do a stats here.
# calculate the correlation between OD index and Orientation preference.
all_cell_info['Sine_Orien'] = np.sin(np.deg2rad(all_cell_info['Best_Orien']))
# Each point graph is here.
plt.cla()
plt.clf()
fig,ax=plt.subplots(figsize = (10,4),dpi = 180)
# sns.boxplot(y='OD_t',x='Best_Orien',data=all_cell_info,ax=ax,hue = 'Location')
sns.boxplot(y='OD_t',x='Best_Orien',data=all_cell_info,ax=ax)
# ax.get_legend().remove()
ax.set_xlabel(f'Orientation Preference')
ax.set_ylabel(f'OD Preference (t)')
ax.set_title(f'Relationship between Eye and Orientation Preference')
# Or plot line?
# sns.lineplot(data = all_cell_info,x = 'Best_Orien',y = 'OD_t',hue = 'Location',legend = None,ax = ax)
#%% stats all group cells.
from scipy.stats import pearsonr,ttest_ind
# #direct ttest almost sig to all group. We need to test effect size.
# orien_groups = all_cell_info.groupby('Best_Orien')
# all_oriens = np.arange(0,180,22.5)
# corr_matrix = pd.DataFrame(-1.0,index = all_oriens,columns= all_oriens)
# for i,c_orien_A in enumerate(all_oriens):
#     c_A_groups = np.array(orien_groups.get_group(c_orien_A)['OD_t'])
#     for j in range(i,len(all_oriens)):
#         c_orien_B = all_oriens[j]
#         c_B_groups = np.array(orien_groups.get_group(c_orien_B)['OD_t'])
#         corr_matrix.loc[c_orien_B,c_orien_A],_ = ttest_ind(c_A_groups,c_B_groups)

pearsonr(all_cell_info['Best_Orien'],all_cell_info['OD_t'])
# sns.regplot(data = all_cell_info,x = 'Best_Orien',y = 'OD_t',x_jitter=5,x_estimator=np.mean)
group = all_cell_info.groupby('Location')
r_list = []
p_list = []
for i,c_loc in enumerate(group):
    c_group = c_loc[1]
    r,p = pearsonr(c_group['Best_Orien'],c_group['OD_t'])
    r_list.append(r)
    p_list.append(p)