
# coding: utf-8

# # Load and preprocess 2008 data
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

file = "../data/interim/2008data.dta"
df_rawest = pd.read_stata(file)


# In[3]:

df_rawest.V085157.value_counts()


# In[4]:

good_columns = [#'campfin_limcorp', # "Should gov be able to limit corporate contributions"
    'V083098x',  # Your own party identification
    
    'V085086',  # Abortion
    'V085139',  # Moral Relativism
    'V085140',  # "Newer" lifetyles
    'V085141',  # Moral tolerance
    'V085142',  # Traditional Families
    'V083211x',  # Gay Job Discrimination
    'V083213',  # Gay Adoption
    'V083212x',  # Gay Military Service
    
    'V083119',  # National health insurance
    'V083128',  # Guaranteed Job
    'V083105',  # Services/Spending
    
#    'V085157',  # Affirmative Action  ( Should this be aapost_hire_x? )
    'V085143', 
    'V085144', 
    'V085145',
    'V085146',
]

df_raw = df_rawest[good_columns]


# ## Clean the data
# ---

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

def dem_edu_special_treatment(x):
    """Eliminate negative numbers and {95. Other}"""
    return np.nan if x == 95 or x <0 else x

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
    "GayMilitaryService",

    "NationalHealthInsurance",
    "StandardOfLiving",
    "ServicesVsSpending",

#    "AffirmativeAction",
    "RacialWorkWayUp",
    "RacialGenerational",
    "RacialDeserve",
    "RacialTryHarder",

    ]
)))

df.Abortion = df.Abortion.apply(lambda x: np.nan if x not in {1, 2, 3, 4} else -x)

df.loc[:, 'NewerLifestyles'] = df.NewerLifestyles.apply(lambda x: -x)  # Tolerance. 1: tolerance, 7: not
df.loc[:, 'TraditionalFamilies'] = df.TraditionalFamilies.apply(lambda x: -x)  # 1: moral relativism, 5: no relativism

df.loc[:, 'ServicesVsSpending'] = df.ServicesVsSpending.apply(lambda x: -x)  # Gov't insurance?

df.loc[:, 'RacialTryHarder'] = df.RacialTryHarder.apply(lambda x: -x)  # Racial support
df.loc[:, 'RacialWorkWayUp'] = df.RacialWorkWayUp.apply(lambda x: -x)  # Systemic factors?


# In[6]:

print("Variables now available: df")


# In[7]:

df_rawest.V083098x.value_counts()


# In[8]:

df.PartyID.value_counts()


# In[9]:

df.describe()


# In[10]:

df.head()


# In[11]:

df.to_csv("../data/processed/2008.csv")


# In[ ]:



