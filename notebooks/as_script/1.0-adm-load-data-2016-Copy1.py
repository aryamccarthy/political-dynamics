
# coding: utf-8

# # Load and preprocess 2016 data
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

file = "../../data/interim/2016data.dta"
df_rawest = pd.read_stata(file)


# In[3]:

good_columns = [#'campfin_limcorp', # "Should gov be able to limit corporate contributions"
    'V161158x',  # Your own party identification
    
    'V161232',  # Abortion
    'V162207',  # Moral Relativism
    'V162208',  # "Newer" lifetyles
    'V162209',  # Moral tolerance
    'V162210',  # Traditional Families
    'V161229x',  # Gay Job Discrimination
    'V161230',  # Gay Adoption
    
    'V161184',  # National health insurance
    'V161189',  # Guaranteed Job
    'V161178',  # Services/Spending

    'V162238x',  
    'V162211',  
    'V162212',
    'V162213',
    'V162214',
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

def negative_to_nan(value):
    """Convert negative values to missing.
    
    ANES codes various non-answers as negative numbers.
    For instance, if a question does not pertain to the 
    respondent.
    """
    return value if value >= 0 else np.nan

def lib1_cons2_neutral3(x):
    """Rearrange questions where 3 is neutral."""
    return -3 + x if x != 1 else x

def liblow_conshigh(x):
    """Reorder questions where the liberal response is low."""
    return -x

df = df_raw.applymap(convert_to_int)
df = df.applymap(negative_to_nan)

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

df.PartyID = df.PartyID.apply(lambda x: np.nan if x == 99 else x)
df.Abortion = df.Abortion.apply(lambda x: np.nan if x not in {1, 2, 3, 4} else -x)

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: x if x != 99 else np.nan)
df.loc[:, 'NationalHealthInsurance'] = df.NationalHealthInsurance.apply(lambda x: x if x != 99 else np.nan)
df.loc[:, 'StandardOfLiving'] = df.StandardOfLiving.apply(lambda x: x if x != 99 else np.nan)


df.loc[:, 'NewerLifestyles'] = df.NewerLifestyles.apply(lambda x: -x)  # Tolerance. 1: tolerance, 7: not
df.loc[:, 'TraditionalFamilies'] = df.TraditionalFamilies.apply(lambda x: -x)  # 1: moral relativism, 5: no relativism

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: -x)  # Gov't insurance?

df.loc[:, 'RacialTryHarder'] = df.RacialTryHarder.apply(lambda x: -x)  # Racial support
df.loc[:, 'RacialWorkWayUp'] = df.RacialWorkWayUp.apply(lambda x: -x)  # Systemic factors?


# In[5]:

print("Variables now available: df")


# In[6]:

df_rawest.V161158x.value_counts()


# In[7]:

df.PartyID.value_counts()


# In[8]:

df.describe()


# In[9]:

df.head()


# In[10]:

df.to_csv("../../data/processed/2016.csv")


# In[13]:

df_rawest.V160102.to_csv("../../data/processed/2016_weights.csv")


# In[ ]:



