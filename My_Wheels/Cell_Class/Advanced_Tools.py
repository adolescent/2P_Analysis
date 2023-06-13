'''
This script will provide advanced data processing method only avaliable on already Z scored data frames.

'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA



def Z_PCA(Z_frame,sample = 'Cell'):
    pca = PCA()
    data = np.array(Z_frame)
    if sample == 'Cell':
        data = data.T# Use cell as sample and frame as feature.
    elif sample == 'Frame':
        data = data
    else:
        raise IOError('Sample method invalid.')
    pca.fit(data)
    PC_Comps = pca.components_# out n_comp*n_feature
    point_coords = pca.transform(data)# in n_sample*n_feature,out n_sample*n_comp
    return PC_Comps,point_coords,pca