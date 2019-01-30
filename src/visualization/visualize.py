import re

from itertools import cycle
from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np

def extract_year(filename: Path) -> str:
    as_string = str(filename)
    match = re.match(r'.*([1-2][0-9]{3}).*', as_string)
    if match is not None:
        return match.group(1)
    else:
        return "0000"


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # From https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return cycle(color_list) # base.from_list(cmap_name, color_list, N)


def biplot(pca, dat, title: str, components=(0, 1), color=None):
    plt.figure()
    pc1, pc2 = components

    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1]
    yvector = pca.components_[pc2]

    tmp = pca.transform(dat.values)
    xs = tmp[:, pc1]
    ys = tmp[:, pc2]


    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)

    color_cycle = discrete_cmap(len(xvector), 'tab20')
    handles, labels = [], []

    plt.xlabel(f"Principal Component {pc1}")
    plt.ylabel(f"Principal Component {pc2}")
    plt.title(title)

    plt.scatter(xs, ys, alpha=0.3, s=40, linewidth=0, c=color, cmap="coolwarm")
    # for i in range(len(xs)):
        # circles project documents (ie rows from csv) as points onto PC axes
        # plt.plot(xs[i], ys[i], 'o', alpha=0.2, markersize=10, color=rgbs[i])
        # plt.text(xs[i]*1.2, ys[i]*1.2, list(dat.index)[i], color='b')

    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        arrow = plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                  color = next(color_cycle),
                  width=0.0005, head_width=0.25)
        handles.append(arrow)
        labels.append(list(dat.columns.values)[i])
        # plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
        #          list(dat.columns.values)[i], color='r')

    plt.legend(handles, labels, fontsize='x-small')

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    ) # labels along the bottom edge are off

    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    ) # labels along the bottom edge are off


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_explained_variance(pca, title: str):
    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)

    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    bottom = range(len(explained_var))
    plt.bar(bottom, explained_var, label='individual explained variance')
    color_cycle = ax._get_lines.prop_cycler
    next(color_cycle)  # Hack to keep both plots from being blue.
    plt.plot(bottom, cum_var_exp, marker='o', label='cumulative explained variance')
    plt.axhline(1/len(explained_var), linestyle='dashed', color='g', label='Kaiser criterion')
    plt.xlabel("Principal components")
    plt.ylabel("Explained variance ratio")

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[2], handles[0], handles[1]]
    labels = [labels[2], labels[0], labels[1]]
    plt.legend(handles, labels)
    if title:
        plt.title(title)

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    ) # labels along the bottom edge are off

    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    ) # labels along the bottom edge are off


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

