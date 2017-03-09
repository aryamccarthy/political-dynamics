from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler


def _do_transform(X, transformer, return_pipeline):
    result = transformer.fit_transform(X)
    if return_pipeline:
        return result, transformer
    else:
        return result


def pca(X, return_pipeline=False):
    imp = Imputer(strategy='mean')
    scl = StandardScaler()
    pca = PCA()
    pipeline = Pipeline([
        ('imp', imp),
        ('scl', scl),
        ('pca', pca),
    ])
    return _do_transform(X, pipeline, return_pipeline)


def scale(X, return_pipeline=False):
    imp = Imputer(strategy='mean')
    scl = StandardScaler()
    pipeline = Pipeline([
        ('imp', imp),
        ('scl', scl),
    ])
    return _do_transform(X, pipeline, return_pipeline)
