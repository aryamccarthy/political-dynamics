"""Plot scatterplots of aspects (Racial, Economic, Moral)."""
import argparse
import logging
import sys


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder)) # Adds higher directory to python modules path.

from sklearn.preprocessing import minmax_scale

from features.build_features import make_pca_and_scaled_data
from visualize import biplot, extract_year

log = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-i", "--infile", type=Path, required=True)
    parser.add_argument("-o", "--outfile", type=Path, required=True)
    return parser.parse_args()

def extract_aspects(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(minmax_scale(data, feature_range=(-1, 1)), columns=data.columns)

    econ = (df.NationalHealthInsurance + df.StandardOfLiving + df.ServicesVsSpending) / 3
    try:
        moral = (df.Abortion + df.MoralRelativism + df.MoralTolerance + df.TraditionalFamilies + df.NewerLifestyles) / 5
    except AttributeError:
        moral = df.Abortion
    try:
        racial = (df.AffirmativeAction + df.RacialDeserve + df.RacialGenerational + df.RacialTryHarder + df.RacialWorkWayUp) / 5
    except AttributeError:
        try:
            racial = (df.RacialDeserve + df.RacialGenerational + df.RacialTryHarder + df.RacialWorkWayUp) / 4
        except AttributeError:
            racial = pd.Series([0] * len(df))
    party = df.PartyID
    aspects = pd.concat([econ, moral, racial, party], axis=1)
    aspects.columns = ["Econ", "Moral", "Racial", "PartyID"]
    return aspects

def map_diag(self, func, **kwargs):
    """
    Seaborn changed this behavior and made it more annoying.
    This is a copy of the original function, before it got annoying.

    Plot with a univariate function on each diagonal subplot.
    Parameters
    ----------
    func : callable plotting function
        Must take an x array as a positional arguments and draw onto the
        "currently active" matplotlib Axes. There is a special case when
        using a ``hue`` variable and ``plt.hist``; the histogram will be
        plotted with stacked bars.
    """
    # Add special diagonal axes for the univariate plot
    if self.square_grid and self.diag_axes is None:
        diag_axes = []
        for i, (var, ax) in enumerate(zip(self.x_vars,
                                          np.diag(self.axes))):
            if i and self.diag_sharey:
                diag_ax = ax._make_twin_axes(sharex=ax,
                                             sharey=diag_axes[0],
                                             frameon=False)
            else:
                diag_ax = ax._make_twin_axes(sharex=ax, frameon=False)
            diag_ax.set_axis_off()
            diag_axes.append(diag_ax)
        self.diag_axes = np.array(diag_axes, np.object)

    # Plot on each of the diagonal axes
    fixed_color = kwargs.pop("color", None)
    for i, var in enumerate(self.x_vars):
        ax = self.diag_axes[i]
        hue_grouped = self.data[var].groupby(self.hue_vals)

        # Special-case plt.hist with stacked bars
        if func is plt.hist:
            plt.sca(ax)

            vals = []
            for label in self.hue_names:
                # Attempt to get data for this level, allowing empty.
                try:
                    vals.append(np.asarray(hue_grouped.get_group(label)))
                except KeyError:
                    vals.append(np.array([]))
            color = self.palette if fixed_color is None else fixed_color

            if "histtype" in kwargs:
                func(vals, color=color, **kwargs)
            else:
                func(vals, color=color, histtype="barstacked", **kwargs)
        else:
            plt.sca(ax)
            for k, label_k in enumerate(self.hue_names):
                # Attempt to get data for this level, allowing empty.
                try:
                    data_k = hue_grouped.get_group(label_k)
                except KeyError:
                    data_k = np.array([])

                if fixed_color is None:
                    color = self.palette[k]
                else:
                    color = fixed_color

                func(data_k, label=label_k, color=color, **kwargs)

        self._clean_axis(ax)

    self._add_axis_labels()

    return self

def main():
    args = parse_args()
    datafile = args.infile
    log.info(f"Reading {datafile}")

    df = pd.read_csv(datafile, index_col=0)

    log.info("Computing aspects")
    aspects = extract_aspects(df)

    log.info("Plotting.")
    # g = sns.pairplot(aspects.sample(100, random_state=1), hue="PartyID", palette=sns.color_palette("RdBu_r", 7), vars=["Econ", "Moral", "Racial"], diag_kind="hist")
    g = sns.PairGrid(aspects.sample(100, random_state=2), hue="PartyID", palette=sns.color_palette("RdBu_r", 7), vars=["Econ", "Moral", "Racial"])
    g.map_offdiag(plt.scatter, alpha=.3, edgecolor="white")
    map_diag(g, plt.hist)

    plt.subplots_adjust(top=0.9)
    year = extract_year(args.infile)
    g.fig.suptitle('Index positions in {}'.format(year))
    g.add_legend()

    destination = args.outfile
    if not destination.suffix == ".pdf":
        destination = destination.with_suffix(".pdf")
    log.info(f"Saving to {destination}")
    plt.savefig(destination, bbox_inches='tight')

if __name__ == '__main__':
    main()
