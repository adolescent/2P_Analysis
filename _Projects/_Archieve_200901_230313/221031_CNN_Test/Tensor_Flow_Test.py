# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:24:56 2022

@author: ZR

Test tensor flow and env setup.

Remember to change env into cnnzr. 
Tensorflow is installed only here.
"""

import tensorflow as tf 

print(tf.__version__)
tf.config.list_physical_devices('GPU')








#%% Torch test here.
import torch
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
