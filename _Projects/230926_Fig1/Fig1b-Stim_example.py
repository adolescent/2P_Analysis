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

work_path = r'D:\_Path_For_Figs\Fig0_Timepoints'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable(expt_folder,'Cell_Class.pkl')

