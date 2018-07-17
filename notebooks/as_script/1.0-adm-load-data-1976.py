
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np


# # Load and preprocess 1976 data
# 
# Time to start looking at other years!
# 
# ---

# In[2]:

import pandas as pd


# ## Load the data.
# 
# ---
# 
# If this fails, be sure that you've saved your own data in the prescribed location, then retry.

# In[3]:

file = "../../data/interim/1976data.dta"  
# Matt Wilson converted the older Stata file to the one we use.
df_rawest = pd.read_stata(file)


# In[4]:

good_columns = [
    # Demographic
    'V763174',  # SUMMARY-R'S PARTY ID
    
    'V763796',  # OPIN:WHEN ALLOW ABORTION (1: never)

    'V763273',  # Private vs public insurance
    'V763241',  # GOVT GUAR JOB/S.L  (1: guarantee)
    'V763353',  # Gov't should spend less, even if cutting health and education.

    'V763264',  # MNRTY GRP AID SCL (1: help)
#    'V763757',  # THE POOR ARE POOR BECAUSE THE AMERICAN WAY OF LIFE DOESN'T GIVE ALL PEOPLE AN EQUAL CHANCE? (1: agree)
]
df_raw = df_rawest[good_columns]


# In[5]:

def convert_to_int(s):
    """Turn ANES data entry into an integer.
    
    >>> convert_to_int("1. Govt should provide many fewer services")
    1
    >>> convert_to_int("2")
    2
    """
    try:
        return int(s.partition('.')[0])
    except ValueError:
        warnings.warn("Couldn't convert: "+s)
        return np.nan
    except AttributeError:
        return s


def not_informative_to_nan(x):
    """Convert non-informative values to missing.
    
    ANES codes various non-answers as 8, 9, and 0.
    For instance, if a question does not pertain to the 
    respondent.
    """
    return np.nan if x in {8, 9, 0} else x


df = df_raw.applymap(convert_to_int)

df.rename(inplace=True, columns=dict(zip(
    good_columns,
    ["PartyID",
    
    "Abortion",
#     "MoralRelativism",
#     "NewerLifestyles",
#     "MoralTolerance",
#     "TraditionalFamilies",
#     "GayJobDiscrimination",
#     "GayMilitaryService",

    "NationalHealthInsurance",
    "StandardOfLiving",
    "ServicesVsSpending",

    "AffirmativeAction",
#     "RacialResentment1",
#     "RacialResentment2",
#     "RacialResentment3",
#     "RacialResentment4",
    ]
)))


non_pid_columns = list(df.columns)
non_pid_columns.remove('PartyID')
df[non_pid_columns] = df[non_pid_columns].applymap(not_informative_to_nan)  # Dropped because its info is different.

# Code so that liberal is lower numbers
df.loc[:, 'PartyID'] = df.PartyID.apply(lambda x: np.nan if x >= 7 else x)  # 7: other minor party, 8: apolitical, 9: NA

df.loc[:, 'Abortion'] = df.Abortion.apply(lambda x: np.nan if x in {7, 8, 9, 0} else -x)

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: -x)


# In[6]:

df.tail()


# In[7]:

print("Variables now available: df")


# In[8]:

df_rawest.V763796.value_counts()


# In[9]:

df.Abortion.value_counts()


# In[10]:

df_rawest.V763174.value_counts()


# In[11]:

df.PartyID.value_counts()


# In[12]:

df.to_csv("../../data/processed/1976.csv")


# In[13]:

df.describe()


# In[19]:

df_rawest.V763353.value_counts()


# In[ ]:



