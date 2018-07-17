
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


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

# Load the "autoreload" extension
# get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
# get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

# import my method from the source code
import features.build_features
import visualization.visualize
from visualization.visualize import biplot, plot_explained_variance, triplot


YEAR = 2004

# In[3]:

df = pd.read_csv("../../data/processed/{}.csv".format(YEAR), index_col=0)


# ---
# 
# ## Correlations in data

# In[4]:

# Spearman is recommended for ordinal data.
correlations = df.corr(method='spearman')
plt.figure()
sns.heatmap(correlations,
           square=True);
# plt.tight_layout()
plt.savefig(f"../../reports/figures/correlations_{YEAR}.pdf", bbox_inches='tight')


# Note that if we were to scale the data, the correlation matrix would be unchanged.

# In[5]:

plt.figure()
cg = sns.clustermap(correlations, square=True)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),
        rotation=0);  # Fix rotation of y-labels.
# plt.tight_layout()
# plt.tight_layout()
plt.savefig(f"../../reports/figures/heatmap_{YEAR}.pdf", bbox_inches='tight')


# There were no strong clusters.
# 
# Abortion was a very separate issue from most.
# 
# Affirmative action and standard of living expressed similar views.

# ## Principal component analysis

# In[6]:

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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
data_pca = pipeline.fit_transform(df)
_scaled = scaler_pipeline.transform(df)


# ### Explained variance
# 
# How much of the variance in the data is explained by each successive component?

# In[7]:

plot_explained_variance(pca, title=f"Explained variance ({YEAR}, impute)")


# ### Biplot
# 
# A scatterplot projected onto the first two principal components.

# In[8]:

data_scaled = pd.DataFrame(_scaled, columns=df.columns)
triplot(pca, data_scaled, title=f'ANES {YEAR} Biplot', color=data_scaled.PartyID)


# In[9]:

biplot(pca, data_scaled, title=f"Biplot ({YEAR}, impute)", color=data_scaled.PartyID)


# In[10]:

pca.explained_variance_


# The party identification isn't even one of the principal axes.

# ## Dropping na

# In[11]:

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
data_pca = pipeline.fit_transform(df2)
_scaled = scaler_pipeline.transform(df2)
data_scaled = pd.DataFrame(_scaled, columns=df.columns)


# In[12]:

biplot(pca, data_scaled, title=f'Biplot ({YEAR}, drop null)', color=data_scaled.PartyID)


# In[14]:

plot_explained_variance(pca, title=f"Explained variance ({YEAR}, drop null)")


# In[ ]:



