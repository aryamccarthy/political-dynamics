
# coding: utf-8

# # Load and preprocess 2004 data
# 
# We will, over time, look over other years. Our current goal is to explore the features of a single year.
# 
# ---

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ## Load the data.
# 
# ---
# 
# If this fails, be sure that you've saved your own data in the prescribed location, then retry.

# In[2]:

file = "../../data/interim/2004data.dta"
df_rawest = pd.read_stata(file)


# In[3]:

good_columns = [
    'V043116',  # Party Identification
    
    'V045132',  #Abortion
    'V045189',  #Moral Relativism
    'V045190',  #“Newer Lifestyles”
    'V045191',  #Moral Tolerance
    'V045192',  #Traditional Families
    'V045156a',  #Gay Job Discrimination
    'V045158',  # Gay adoption
    'V045157a',  # Gay military service
    
    'V043150',  #National Health Insurance
    'V043152',  #Guaranteed Job
    'V045121',  #Services/Spending
    
    'V045207a',  #Affirmative Action
    'V045193',  #Racial Resentment 1
    'V045194',  #Racial Resentment2
    'V045195',  #Racial Resentment3
    'V045196',  #Racial Resentment4
]
df_raw = df_rawest[good_columns]


# ## Clean the data
# ---

# In[4]:

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
    "MoralRelativism",
    "NewerLifestyles",
    "MoralTolerance",
    "TraditionalFamilies",
    "GayJobDiscrimination",
    "GayAdoption",
    "GayMilitaryService",

    "NationalHealthInsurance",
    "StandardOfLiving",
    "ServicesVsSpending",

    "AffirmativeAction",
    "RacialWorkWayUp",
    "RacialGenerational",
    "RacialDeserve",
    "RacialTryHarder",
    ]
)))

non_pid_columns = list(df.columns)
non_pid_columns.remove('PartyID')
df[non_pid_columns] = df[non_pid_columns].applymap(not_informative_to_nan)

# Code so that liberal is lower numbers
df.loc[:, 'PartyID'] = df.PartyID.apply(lambda x: np.nan if x >= 7 else x)  # 7: other minor party, 8: apolitical, 9: NA

df.loc[:, 'Abortion'] = df.Abortion.apply(lambda x: np.nan if x in {7, 8, 9, 0} else -x)


df.loc[:, 'NewerLifestyles'] = df.NewerLifestyles.apply(lambda x: -x)  # Tolerance. 1: tolerance, 7: not
df.loc[:, 'TraditionalFamilies'] = df.TraditionalFamilies.apply(lambda x: -x)  # 1: moral relativism, 5: no relativism

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: np.nan if x > 10 else x)
df.loc[:, 'NationalHealthInsurance'] = df.NationalHealthInsurance.apply(lambda x: np.nan if x > 10 else x)
df.loc[:, 'StandardOfLiving'] = df.StandardOfLiving.apply(lambda x: np.nan if x > 10 else x)



df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: -x)  # Gov't insurance?

df.loc[:, 'RacialTryHarder'] = df.RacialTryHarder.apply(lambda x: -x)  # Racial support
df.loc[:, 'RacialWorkWayUp'] = df.RacialWorkWayUp.apply(lambda x: -x)  # Systemic factors?


# In[5]:

print("Variables now available: df")


# In[6]:

df_rawest.V043116.value_counts()


# In[7]:

df.PartyID.value_counts()


# In[8]:

df.head()


# In[9]:

df.describe()


# In[10]:

df.to_csv("../../data/processed/2004.csv")


# In[14]:

df_rawest.V040102.to_csv("../../data/processed/2004_weights.csv")


# In[ ]:



