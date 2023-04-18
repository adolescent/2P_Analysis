'''
This script will plot umap results.

Notice that for plot, fit() not fit_transform() is required. So you need to transform manually if you want to plot by hand.

'''

#%% Import and load

import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import OS_Tools_Kit as ot
from scipy.io import arff
#%%
# pendigits = sklearn.datasets.load_digits()
pendigits = ot.Load_Variable('pendigits.pkl')
# mnist = sklearn.datasets.fetch_openml('mnist_784')
# fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')
mnist = arff.loadarff(f'mnist_784.arff')
fmnist = arff.loadarff(f'Fashion-MNIST.arff')
mnist_show = pd.DataFrame(mnist[0])


#%% Plot graphs.
mapper = umap.UMAP().fit(pendigits.data)# this is only a fit, not transform.Use mapper.transfomr(pendigits.data) to get previous data.
import umap.plot
plt.switch_backend('webAgg')
plt.clf()
# umap.plot.points(mapper)
# plt.show()
umap.plot.points(mapper, labels=pendigits.target)# order of data is the same, so we can use manually defined label.
plt.show()

