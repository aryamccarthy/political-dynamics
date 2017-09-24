
# coding: utf-8

# In[1]:

get_ipython().magic('pylab --no-import-all inline')


# # Factor analysis

# In[2]:

import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer


# In[3]:

df = pd.read_csv("../data/processed/1976.csv")


# In[28]:

def factor_loadings(data, n_components):
    fa = FactorAnalysis(n_components=n_components)
    scl = StandardScaler()
    imp = Imputer()
    pipeline = Pipeline([('imp', imp), ('scl', scl), ('fa', fa)])
    pipeline.fit(data)
    return pd.DataFrame(fa.components_, columns=data.columns)


# In[29]:

scl = StandardScaler()
imp = Imputer()
pipeline = Pipeline([('imp', imp), ('scl', scl)])
pipeline.fit_transform(df)


# In[30]:

factor_loadings(df, 3)


# In[ ]:



