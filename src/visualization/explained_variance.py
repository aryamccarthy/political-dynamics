"""Plot PCA explained variance."""
import argparse
import logging
import sys


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder)) # Adds higher directory to python modules path.

from features.build_features import make_pca_and_scaled_data
from visualize import plot_explained_variance, extract_year

log = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO)

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument("-i", "--infile", type=Path, required=True)
	parser.add_argument("-o", "--outfile", type=Path, required=True)
	parser.add_argument("-s", "--strategy", type=str, choices={"impute", "drop"}, default="drop")
	return parser.parse_args()

def main():
	args = parse_args()
	datafile = args.infile
	log.info(f"Reading {datafile}")

	df = pd.read_csv(datafile, index_col=0)

	log.info("Computing and plotting variances")
	scaled_data, pca = make_pca_and_scaled_data(df, missing_strategy=args.strategy)

	# Info for figure title.
	explanation = args.strategy + " null"
	year = extract_year(args.infile)

	plot_explained_variance(pca, title=f"Explained variance ({year}, {explanation})")

	# Saving the figure.
	destination = args.outfile
	if not destination.suffix == ".pdf":
		destination = destination.with_suffix(".pdf")
	log.info(f"Saving to {destination}")
	plt.savefig(destination, bbox_inches='tight')

if __name__ == '__main__':
	main()
