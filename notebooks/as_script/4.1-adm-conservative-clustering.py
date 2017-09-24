
# coding: utf-8

# In[1]:

get_ipython().magic('run 1.0-adm-load-data-2012.ipynb')


# In[2]:

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


# In[3]:

dem = df[data_pca[:, 0] <= 0]
rep = df[data_pca[:, 0] > 0]


# In[4]:

rep_pca = pipeline.fit_transform(rep)
rep_scaled = scaler_pipeline.transform(rep)


# In[5]:

def plot_explained_variance(pca):
    import plotly
    from plotly.graph_objs import Scatter, Marker, Layout, XAxis, YAxis, Bar, Line
    plotly.offline.init_notebook_mode() # run at the start of every notebook
    
    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)
    
    plotly.offline.iplot({
        "data": [Bar(y=explained_var, name='individual explained variance'),
                 Scatter(y=cum_var_exp, name='cumulative explained variance')
            ],
        "layout": Layout(xaxis=XAxis(title='Principal components'), yaxis=YAxis(title='Explained variance ratio'))
    })

plot_explained_variance(pca)


# In[9]:

def biplot(pca, dat, title='', show_points=True, components=(0, 1)):
    import plotly
    from plotly.graph_objs import Scatter, Marker, Layout, XAxis, YAxis, Bar, Line
    plotly.offline.init_notebook_mode() # run at the start of every notebook

    pc1, pc2 = components
    
    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1] 
    yvector = pca.components_[pc2]

    tmp = pca.transform(dat.values)
    xs = tmp[:,pc1] 
    ys = tmp[:,pc2]
    if show_points:
        annotations = [Scatter(x=xs, y=ys, mode ='markers', marker=dict(size=1), name='cumulative explained variance')]
    else:
        annotations = []
    for i in range(len(xvector)):
        txt = list(dat.columns.values)[i]
        annotations.append(
                Scatter(
                    x=[0, xvector[i]*max(xs)],
                    y=[0, yvector[i]*max(ys)],
                    mode='lines+text',
                    text=['', txt],
                    name=txt,
                ))
    
    plotly.offline.iplot({
        "data": annotations,
        "layout": Layout(xaxis=XAxis(title='Principal Component ' + str(pc1 + 1)), 
                         yaxis=YAxis(title='Principal Component ' + str(pc2 + 1)),
                        title=title)
    })


    plt.show()
biplot(pca, pd.DataFrame(rep_scaled, columns=df.columns), title='Biplot for conservatives', components=(0, 1))


# In[8]:

rep.mean()


# In[ ]:



