
# coding: utf-8

# # Load and preprocess 1992 data
# 
# We will, over time, look over other years. Our current goal is to explore the features of a single year.
# 
# ---

# In[1]:

get_ipython().magic('pylab --no-import-all inline')
import pandas as pd


# ## Load the data.
# 
# ---
# 
# If this fails, be sure that you've saved your own data in the prescribed location, then retry.

# In[2]:

file = "../data/interim/1992data.dta"
df_rawest = pd.read_stata(file)


# In[ ]:




# In[3]:

good_columns = [#'campfin_limcorp', # "Should gov be able to limit corporate contributions"
    'V923634',  # Your own party identification
    
    'V923732',  # Abortion
    'V926115',  # Moral Relativism
    'V926118',  # "Newer" lifetyles
    'V926116',  # Moral tolerance
    'V926117',  # Traditional Families
    'V925924',  # Gay Job Discrimination
    'V925928',  # Gay Adoption
    'V925926',  # Gay Military Service
    
    'V923716',  # National health insurance
    'V923718',  # Guaranteed Job
    'V923701',  # Services/Spending
    
    'V900464',  # Affirmative Action -- 1-5; 7 is other
    'V926126', 
    'V926129', 
    'V926127',
    'V926128',
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

def not_informative_to_nan(x):
    """Convert non-informative values to missing.
    
    ANES codes various non-answers as 8, 9, and 0.
    For instance, if a question does not pertain to the 
    respondent.
    """
    return np.nan if x in {8, 9, 0} else x


df = df_raw.applymap(convert_to_int)
non_pid_columns = list(df.columns)
non_pid_columns.remove('V923634')
df[non_pid_columns] = df[non_pid_columns].applymap(not_informative_to_nan)


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

# Code so that liberal is lower numbers
df.loc[:, 'PartyID'] = df.PartyID.apply(lambda x: np.nan if x >= 7 else x)  # 7: other minor party, 8: apolitical, 9: NA

df.loc[:, 'Abortion'] = df.Abortion.apply(lambda x: np.nan if x in {7, 8, 9, 0} else -x)


df.loc[:, 'NewerLifestyles'] = df.NewerLifestyles.apply(lambda x: -x)  # Tolerance. 1: tolerance, 7: not
df.loc[:, 'TraditionalFamilies'] = df.TraditionalFamilies.apply(lambda x: -x)  # 1: moral relativism, 5: no relativism

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: -x)  # Gov't insurance?

df.loc[:, 'RacialTryHarder'] = df.RacialTryHarder.apply(lambda x: -x)  # Racial support
df.loc[:, 'RacialWorkWayUp'] = df.RacialWorkWayUp.apply(lambda x: -x)  # Systemic factors?


# In[5]:

print("Variables now available: df")


# In[6]:

df_rawest.V923634.value_counts()


# In[7]:

df.PartyID.value_counts()


# In[8]:

df.head()


# In[9]:

df.to_csv("../data/processed/1992.csv")


# In[ ]:



