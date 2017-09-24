
# coding: utf-8

# In[1]:

get_ipython().magic('pylab --no-import-all inline')


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
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.pardir, 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport visualization.visualize')
from visualization.visualize import biplot, plot_explained_variance, triplot


# In[3]:

df = pd.read_csv("../data/processed/2012.csv", index_col=0)


# ---
# 
# ## Correlations in data

# In[4]:

# Spearman is recommended for ordinal data.
correlations = df.corr(method='spearman')
sns.heatmap(correlations, square=True);


# Note that if we were to scale the data, the correlation matrix would be unchanged.

# In[5]:

cg = sns.clustermap(correlations, square=True)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),
        rotation=0);  # Fix rotation of y-labels.


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

plot_explained_variance(pca)


# In[8]:

pca.explained_variance_


# In[9]:

pd.DataFrame(_scaled, columns=df.columns).head()


# ### Biplot
# 
# A scatterplot projected onto the first two principal components.

# In[10]:

plt.figure()
data_scaled = pd.DataFrame(_scaled, columns=df.columns)
triplot(pca, data_scaled, title='ANES 2012 Biplot', color=data_scaled.PartyID)


# In[11]:

biplot(pca, data_scaled, title='ANES 2012 Biplot', color=data_scaled.PartyID)


# Sure, all of the original axes are negative in the first component. That's okay! To quote Dr. Eric Larson: 
# > Because all the data is somewhat correlated, giving a mostly unidimensional representation. Positive/negative isn't so important because eigenvectors could theoretically start anywhere--but traditionally we use the origin.
# 
# **Update:** The demographic factor of education level has a different sign from the others.

# In[12]:

def fpc_ordered(corr):
    """Reorder correlation matrix based on first principal component (FPC)."""
    ew, ev = np.linalg.eig(corr)
    idx = np.argsort(ew)[::-1]  # Reordering index of eigenvalues
    ew, ev = ew[idx], ev[:, idx]
    e1 = ev[:, 0]
    order = np.argsort(e1)
    try:
        return corr.values[:, order][order]
    except AttributeError:
        return corr[:, order][order]
sns.heatmap(fpc_ordered(correlations),
           square=True)
plt.title("FPC Ordering");


# ### Norris ordering
# 
# Arrange correlation matrix by sum of principal components (left singular vectors), weighted by singular values.

# In[13]:

U, S, V = PCA()._fit(X=_scaled)
np.allclose(U @ np.diag(S) @ V, _scaled)


# We see something important here: normally, the singular value decomposition is $\mathbf{A} = \mathbf{U\Sigma V^T}$ where the columns of $\mathbf{V}$ are the left singular vectors. Here, we see that $\mathbf{A} = \mathbf{U\Sigma V}$. Consequently, the rows of $\mathbf{V}$ are the left singular vectors.

# In[14]:

weights = np.abs(S @ V)  # Total weights of each scaled component
norris_ordering = np.argsort(weights)[::-1]
sns.heatmap(correlations.values[:, norris_ordering][norris_ordering],
           square=True)
plt.title("Weighted PCA (Norris) Ordering");


# In[15]:

import rpy2.ipython


# In[16]:

scaled = pd.DataFrame(_scaled, columns=df.columns)


# In[17]:

get_ipython().magic('load_ext rpy2.ipython')


# In[18]:

get_ipython().run_cell_magic('R', '-i scaled', 'library(corrplot)\nM <- cor(scaled)\n(order.AOE <- corrMatOrder(M, order = "AOE"))\n(order.FPC <- corrMatOrder(M, order = "FPC"))\nM.AOE <- M[order.AOE,order.AOE]\nM.FPC <- M[order.FPC,order.FPC]\ncorrplot(M.AOE)\ncorrplot(M.FPC)\ncorrplot(M, order = "hclust", addrect = 4)')


# ## Grouping by largest corresponding principal component
# 
# PCA transforms data of shape `(n_samples, n_features)` into `(n_samples, n_components)` where `n_components ≤ n_features`. The inverse transform, then, converts data of shape `(m, n_components)` to `(m, n_features)`. We want to determine the weights of each feature in the original space. Transforming the identity matrix into the *singular space* would give us the principal components; conversely, transforming the identity matrix *from* the singular space gives us the weights of the basis vectors in each principal component.

# In[19]:

N_COMPONENTS = 3
pca_inverse = np.abs(pca.inverse_transform(np.eye(pca.n_components_)))
most_contributing_component = np.argmax(pca_inverse[:,:N_COMPONENTS], axis=1)
print(np.bincount(most_contributing_component))


# In[20]:

ordering = np.argsort(most_contributing_component)
print(np.sort(most_contributing_component))


# In[21]:

sns.heatmap(correlations.values[:, ordering][ordering],
           square=True);


# In[ ]:



