
# coding: utf-8

# In[ ]:

get_ipython().magic('pylab --no-import-all inline')


# # Exploratory data analysis
# 
# Finding the odd groupings of the data.
# 
# ---

# In[7]:

import pandas as pd


# In[8]:

df = pd.read_csv("../data/processed/2012.csv", index_col=0)


# ---
# 
# ## Summary of the data

# ### Shape

# In[9]:

n_rows, n_cols = df.shape
print("We are analysing {} respondents' answers to {} questions.".format(n_rows, n_cols))


# ### Column names

# In[10]:

df.columns


# ### Data types
# 
# They're all floats to include the missing value, `np.nan`.

# In[11]:

df.dtypes


# ### Missing values by column

# In[12]:

df.isnull().sum()


# ## Visualization

# In[13]:

import seaborn as sns


# In[14]:

sns.boxplot(data=df, orient='h');


# In[15]:

sns.countplot(hue=df.pid_self, x=df.relig_churchoft, palette="RdBu");


# In[16]:

sns.countplot(df.dem_edu.dropna());


# The education distribution isn't pretty. It's bimodal. The two big lumps are:
# 
# * 9 and 10: High school graduate or some college
# * 13: Bachelor's degree
# 
# Since our imputation strategy relies on the assumption of a normal distribution, we're in some hot water here.

# In[17]:

sns.countplot(df.campfin_limcorp.dropna());


# In[ ]:



