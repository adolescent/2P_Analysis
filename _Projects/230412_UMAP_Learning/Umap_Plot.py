'''
This script will plot umap results.
'''

#%% Import and load

import sklearn.datasets
import pandas as pd
import numpy as np
import umap
pendigits = sklearn.datasets.load_digits()
mnist = sklearn.datasets.fetch_openml('mnist_784')
fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')

#%%

