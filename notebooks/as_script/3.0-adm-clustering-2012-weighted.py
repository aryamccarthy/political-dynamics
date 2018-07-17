
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

# # Clustering analysis
# 
# Use various clustering techniques to identify a good subset of questions.
# 
# ---

# In[2]:

import os
import sys

import pandas as pd
import seaborn as sns


# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

# import my method from the source code
import features.build_features
import visualization.visualize
from visualization.visualize import biplot, plot_explained_variance, triplot


# In[3]:

YEAR = 2012


# In[4]:

df = pd.read_csv("../data/processed/{}.csv".format(YEAR), index_col=0)


# In[5]:

weights = pd.read_csv("../data/processed/{}_weights.csv".format(YEAR), index_col=0, header=None, squeeze=True)
weights.shape


# In[6]:

sns.distplot(weights)


# ## Principal component analysis

# In[7]:

def make_weights_matrix(weights, X):
    if weights.shape[0] != X.shape[0]:
        raise ValueError("weights {} and X {} must have same length.".format(weights.shape, X.shape))
    w_new = np.empty_like(X)
    w_new[:] = weights[:, np.newaxis]
    return w_new
w = make_weights_matrix(weights, df)


# In[8]:

w.shape, df.shape


# In[9]:

from wpca import WPCA as PCA


# In[10]:

from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler

imp = Imputer(strategy='mean')
scl = StandardScaler()
pca = PCA()
pipeline = Pipeline([
        ('imp', imp),
        ('scl', scl),
        ('pca', pca),
    ])
scaler_pipeline = Pipeline([
        ('imp', imp),
        ('scl', scl),
    ])
data_pca = pipeline.fit_transform(df, pca__weights=w)
_scaled = scaler_pipeline.transform(df)


# ### Explained variance
# 
# How much of the variance in the data is explained by each successive component?

# In[11]:

plot_explained_variance(pca)


# ### Biplot
# 
# A scatterplot projected onto the first two principal components.

# In[12]:

data_scaled = pd.DataFrame(_scaled, columns=df.columns)
triplot(pca, data_scaled, title='ANES {} Biplot'.format(YEAR), color=data_scaled.PartyID)


# In[13]:

biplot(pca, data_scaled, title='ANES {} Biplot'.format(YEAR), color=data_scaled.PartyID)


# In[14]:

pca.explained_variance_


# ## Dropping na

# In[15]:

df2 = df.dropna()
#imp = Imputer(strategy='mean')
scl = StandardScaler()
pca = PCA()
pipeline = Pipeline([
#        ('imp', imp),
        ('scl', scl),
        ('pca', pca),
    ])
scaler_pipeline = Pipeline([
#        ('imp', imp),
        ('scl', scl),
    ])
data_pca = pipeline.fit_transform(df2, pca__weights=w[df2.index])
_scaled = scaler_pipeline.transform(df2)
data_scaled = pd.DataFrame(_scaled, columns=df.columns)


# In[16]:

biplot(pca, data_scaled, title='ANES {} Biplot'.format(YEAR), color=data_scaled.PartyID)


# In[17]:

plot_explained_variance(pca)


# In[ ]:



