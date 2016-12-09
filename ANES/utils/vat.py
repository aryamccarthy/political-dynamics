"""
VAT: visual assessment of cluster tendency

Arya McCarthy

Reimplementing the code of Michael Hahsler et al.
"""

import matplotlib.pyplot as plt
import numpy as np

dt = np.float64


def seriate_dist_VAT(x):
    D = np.copy(x)
    N = D.shape[0]

    P = [0 for _ in range(N)]
    I = np.array([False for _ in range(N)])
    J = np.array([True for _ in range(N)])

    i, _ = np.unravel_index(np.argmax(D), np.shape(D))

    P[0] = i
    I[i] = True
    J[i] = False

    for r in range(1, N):
        D2 = D[np.where(I)[0]][:, np.where(J)[0]]
        _, j = np.unravel_index(np.argmin(D2), np.shape(D2))
        j = np.where(J)[0][j]
        P[r] = j
        I[j] = True
        J[j] = False

    return P

def pathdist_floyd(x):
    """Calculate the path distance for iVAT."""
    y = np.copy(x)
    n = np.shape(y)[0]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if np.maximum(y[k, i], y[j, k]) < y[j, i]:
                    y[j, i] = np.maximum(y[k, i], y[j, k])
    return y


def path_dist(x):
    """Calculate path distance from iVAT using a modified Floyd's alg."""
    m = np.asarray(x, dtype=dt)
    if np.any(np.isnan(m)):
        raise RuntimeError("nans not allowed in x.")

    m[np.isinf(m)] = np.finfo(dt).max

    if np.any(m < 0):
        raise RuntimeError("Negative values not allowed in x.")

    m = pathdist_floyd(m)


def vat(x, **kwargs):
    r, c = x.shape
    if r != c:
        raise RuntimeError("Must use a (square) distance matrix.")
    new_order = seriate_dist_VAT(x)
    plt.spy(x[:, new_order][new_order])


def ivat(x, **kwargs):
    r, c = x.shape
    if r != c:
        raise RuntimeError("Must use a (square) distance matrix.")
    x = path_dist(x)
    new_order = seriate_dist_VAT(x)
    plt.spy(x[:, new_order][new_order])
