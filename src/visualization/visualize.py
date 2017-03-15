import numpy as np
from plotly.graph_objs import Scatter, Layout, XAxis, YAxis, Bar, Scatter3d
import plotly.offline as py

py.init_notebook_mode()  # run at the start of every notebook


def plot_explained_variance(pca):
    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)

    py.iplot({
        "data": [
            Bar(y=explained_var, name='individual explained variance'),
            Scatter(y=cum_var_exp, name='cumulative explained variance')
        ],
        "layout":
            Layout(xaxis=XAxis(title='Principal components'),
                   yaxis=YAxis(title='Explained variance ratio')
                   )
    })


def biplot(pca, dat, title='', components=(0, 1), color=None):
    pc1, pc2 = components

    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1]
    yvector = pca.components_[pc2]

    tmp = pca.transform(dat.values)
    xs = tmp[:, pc1]
    ys = tmp[:, pc2]

    annotations = [
        Scatter(x=xs, y=ys, mode='markers',
                name='cumulative explained variance')
    ]
    for i in range(len(xvector)):
        txt = list(dat.columns.values)[i]
        annotations.append(
            Scatter(
                x=[0, xvector[i] * max(xs)],
                y=[0, yvector[i] * max(ys)],
                mode='lines+text',
                marker=dict(color=color),
                text=['', txt],
                name=txt,
            ))
    py.iplot({
        "data": annotations,
        "layout": Layout(xaxis=XAxis(title='Principal Component One'),
                         yaxis=YAxis(title='Principal Component Two'),
                         title=title)
    })


def triplot(pca, dat, title='', components=(0, 1, 2), color=None):
    pc1, pc2, pc3 = components

    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1]
    yvector = pca.components_[pc2]
    zvector = pca.components_[pc3]

    tmp = pca.transform(dat.values)
    xs = tmp[:, pc1]
    ys = tmp[:, pc2]
    zs = tmp[:, pc3]

    annotations = [
        Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                  marker=dict(size=2, opacity=0.2, color=color),
                  name='cumulative explained variance')
    ]
    for i in range(len(xvector)):
        txt = list(dat.columns.values)[i]
        annotations.append(
            Scatter3d(
                x=[0, xvector[i] * max(xs)],
                y=[0, yvector[i] * max(ys)],
                z=[0, zvector[i] * max(zs)],
                mode='lines+text',
                text=['', txt],
                name=txt,
            ))
    py.iplot({
        "data": annotations,
        "layout": Layout(xaxis=XAxis(title='Principal Component One'),
                         yaxis=YAxis(title='Principal Component Two'),
                         title=title)
    })
