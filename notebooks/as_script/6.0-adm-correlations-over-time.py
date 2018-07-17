
# coding: utf-8

# In[1]:

# get_ipython().magic('pylab --no-import-all inline')
import matplotlib.pyplot as plt
import numpy as np


# # Correlations over time

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
# import visualization.visualize
# from visualization.visualize import biplot, plot_explained_variance, triplot


# In[3]:

YEARS = [1976, 1988, 1992, 1996, 2000, 2004, 2008, 2012]


# In[4]:

DATA_FRAMES = [pd.read_csv("../../data/processed/{year}.csv".format(year=year), index_col=0) for year in YEARS]


# In[5]:

def plot_correlation_with_party_over_time(var_name):
    years_available = YEARS[:]  # Copy
    corrs = []
    for df, year in zip(DATA_FRAMES, years_available[:]):
        try: 
            corrs.append(df.PartyID.corr(df[var_name], method='spearman'))
        except KeyError:
            years_available.remove(year)
    plt.plot(years_available, corrs, label=var_name)
    plt.title(var_name)
    plt.ylabel("Correlation with Party ID")
    plt.xlabel("Year")
    plt.xlim([1975, 2017])
    plt.ylim(0, 0.6)


def plot_and_close(var_name):
    plot_correlation_with_party_over_time(var_name)
    plt.savefig(f"../../reports/figures/correlation_over_time_{var_name}.pdf")
    plt.close()


# In[6]:

sns.set_palette("Set1", n_colors=20)
from functools import reduce
all_variables_list = [set(df.columns) for df in DATA_FRAMES[1:]]
VARIABLES_IN_ANY_YEAR = list(reduce(set.union, all_variables_list))
for var in VARIABLES_IN_ANY_YEAR:
    plot_correlation_with_party_over_time(var)
plt.legend()
plt.title("");
plt.savefig("../../reports/figures/everything.pdf")
plt.close()


# In[7]:

plot_and_close("Abortion")


# In[8]:

plot_and_close("NationalHealthInsurance")


# In[9]:

plot_and_close("StandardOfLiving")


# In[10]:

plot_and_close("ServicesVsSpending")


# In[11]:

plot_and_close("AffirmativeAction")


# ## And then Trump happened.

# In[12]:

YEARS.append(2016)


# In[13]:

DATA_FRAMES = [pd.read_csv("../../data/processed/{year}.csv".format(year=year), index_col=0) for year in YEARS]


# In[14]:

from functools import reduce
all_variables_list = [set(df.columns) for df in DATA_FRAMES[1:]]
VARIABLES_CONSISTENT_ACROSS_ALL_YEARS = list(reduce(set.intersection, all_variables_list))


# In[15]:

sns.set_palette("Set1", n_colors=20)
from functools import reduce
all_variables_list = [set(df.columns) for df in DATA_FRAMES[1:]]
VARIABLES_IN_ANY_YEAR = list(reduce(set.union, all_variables_list))
for var in VARIABLES_IN_ANY_YEAR:
    plot_correlation_with_party_over_time(var)
plt.legend()
plt.title("");
plt.savefig("../../reports/figures/everything+2016.pdf")
plt.close()


# In[16]:

sns.set_palette("Set1", n_colors=20)
from functools import reduce
all_variables_list = [set(df.columns) for df in DATA_FRAMES[1:]]
VARIABLES_IN_ANY_YEAR = list(reduce(set.union, all_variables_list))
for var in VARIABLES_IN_ANY_YEAR:
    if var.startswith("Racial"):
        plot_correlation_with_party_over_time(var)
plt.legend()
plt.title("");
plt.savefig("../../reports/figures/racial.pdf")


# In[17]:

plot_and_close("Abortion")


# In[18]:

plot_and_close("NationalHealthInsurance")


# In[19]:

plot_and_close("StandardOfLiving")


# In[20]:

plot_and_close("ServicesVsSpending")


# In[21]:

plot_and_close("GayAdoption")


# In[22]:

plot_and_close("GayMilitaryService")


# In[23]:

plot_and_close("MoralRelativism")


# In[24]:

plot_and_close("NewerLifestyles")


# In[25]:

plot_and_close("MoralTolerance")


# In[26]:

plot_and_close("TraditionalFamilies")


# In[27]:

plot_and_close("AffirmativeAction")


# In[28]:

plot_and_close("RacialTryHarder")


# In[29]:

plot_and_close("RacialDeserve")


# In[30]:

plot_and_close("RacialGenerational")


# In[31]:

plot_and_close("RacialWorkWayUp")


# In[ ]:



