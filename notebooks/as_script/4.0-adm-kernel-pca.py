
# coding: utf-8

# In[1]:

get_ipython().magic('pylab --no-import-all inline')


# # Kernel PCA exploration

# In[18]:

import os
import sys

import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.pardir, 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport visualization.visualize')
from visualization.visualize import biplot, triplot, plot_explained_variance


# In[5]:

df = pd.read_csv("../data/processed/2012.csv", index_col=0)


# In[6]:

scaler = Pipeline([
        ('imp', Imputer(strategy='mean')),
        ('scl', StandardScaler())
    ])
X = scaler.fit_transform(df)


# ## Kernel PCA with radial basis function as kernel

# In[8]:

kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)


# In[14]:

pca = PCA()
X_pca = pca.fit_transform(X)


# In[5]:

# Plot results.

plt.figure()
# plt.subplot(4, 1, 1, aspect='equal')
plt.title("Original space")
y = df_raw.pid_self# y = X_pca[:, 0] < 0
reds = y == 1
blues = y == -1
greens = y == 0

plt.plot(X[reds, 0], X[reds, 1], "r,")
plt.plot(X[blues, 0], X[blues, 1], "b,")
plt.plot(X[greens, 0], X[greens, 1], "g,")


plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
# X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# # projection on the first principal component (in the phi space)
# Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
# plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.figure() # plt.subplot(4, 1, 2, aspect='equal')
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "r,")
plt.plot(X_pca[blues, 0], X_pca[blues, 1], "b,")
plt.plot(X_pca[greens, 0], X_pca[greens, 1], "g,")

plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 3, aspect='equal')
plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "r,")
plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "b,")
plt.plot(X_kpca[greens, 0], X_kpca[greens, 1], "g,")

plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 4, aspect='equal')
plt.plot(X_back[reds, 0], X_back[reds, 1], "r,")
plt.plot(X_back[blues, 0], X_back[blues, 1], "b,")
plt.plot(X_back[greens, 0], X_back[greens, 1], "g,")

plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# In[6]:

X_kpca.shape


# ## Kernel PCA with cubic polynomial as kernel

# In[7]:

poly = KernelPCA(kernel='poly', fit_inverse_transform=True, gamma=10)
X_poly = poly.fit_transform(X)
X_pback = poly.inverse_transform(X_poly)


# In[8]:

# Plot results.

plt.figure()
# plt.subplot(4, 1, 1, aspect='equal')
plt.title("Original space")
y = X_pca[:, 0] >= 0
reds = y == True
blues = y == False

plt.plot(X[reds, 0], X[reds, 1], "r.")
plt.plot(X[blues, 0], X[blues, 1], "b.")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
# X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# # projection on the first principal component (in the phi space)
# Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
# plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.figure() # plt.subplot(4, 1, 2, aspect='equal')
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "r,")
plt.plot(X_pca[blues, 0], X_pca[blues, 1], "b,")
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 3, aspect='equal')
plt.plot(X_poly[reds, 0], X_poly[reds, 1], "r.")
plt.plot(X_poly[blues, 0], X_poly[blues, 1], "b.")
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 4, aspect='equal')
plt.plot(X_pback[reds, 0], X_pback[reds, 1], "r.")
plt.plot(X_pback[blues, 0], X_pback[blues, 1], "b.")
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# In[9]:

poly = KernelPCA(kernel='poly', fit_inverse_transform=True)
X_poly = poly.fit_transform(X)
X_pback = poly.inverse_transform(X_poly)


# In[10]:

# Plot results.

plt.figure()
# plt.subplot(4, 1, 1, aspect='equal')
plt.title("Original space")
y = X_pca[:, 0] >= 0
reds = y == True
blues = y == False

plt.plot(X[reds, 0], X[reds, 1], "r.")
plt.plot(X[blues, 0], X[blues, 1], "b.")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
# X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# # projection on the first principal component (in the phi space)
# Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
# plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.figure() # plt.subplot(4, 1, 2, aspect='equal')
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "r,")
plt.plot(X_pca[blues, 0], X_pca[blues, 1], "b,")
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 3, aspect='equal')
plt.plot(X_poly[reds, 0], X_poly[reds, 1], "r.")
plt.plot(X_poly[blues, 0], X_poly[blues, 1], "b.")
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 4, aspect='equal')
plt.plot(X_pback[reds, 0], X_pback[reds, 1], "r.")
plt.plot(X_pback[blues, 0], X_pback[blues, 1], "b.")
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# ## $k$-Means clustering

# In[11]:

from sklearn.cluster import KMeans


# In[12]:

kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(X)
pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)


# In[13]:

plt.figure()
# plt.subplot(4, 1, 1, aspect='equal')
plt.title("Original space")
y = X_pca[:, 0] >= 0
reds = kmeans.labels_ == 0
blues = kmeans.labels_ == 1

plt.plot(X[reds, 0], X[reds, 1], "r.")
plt.plot(X[blues, 0], X[blues, 1], "b.")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
# X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# # projection on the first principal component (in the phi space)
# Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
# plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.figure() # plt.subplot(4, 1, 2, aspect='equal')
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "r,")
plt.plot(X_pca[blues, 0], X_pca[blues, 1], "b,")
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 3, aspect='equal')
plt.plot(X_poly[reds, 0], X_poly[reds, 1], "r.")
plt.plot(X_poly[blues, 0], X_poly[blues, 1], "b.")
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.figure() # plt.subplot(4, 1, 4, aspect='equal')
plt.plot(X_pback[reds, 0], X_pback[reds, 1], "r.")
plt.plot(X_pback[blues, 0], X_pback[blues, 1], "b.")
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# In[14]:

df.pid_self.value_counts()


# In[15]:

df.postvote_presvtwho.value_counts()


# In[19]:

data = pd.DataFrame(X, columns=df.columns)
triplot(pca, data)


# In[ ]:



