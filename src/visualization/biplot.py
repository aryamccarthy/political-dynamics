"""Plot PCA axes."""
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
from visualize import biplot, extract_year

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

	log.info("Creating biplot")
	scaled_data, pca = make_pca_and_scaled_data(df, missing_strategy=args.strategy)

	explanation = args.strategy + " null"
	year = extract_year(args.infile)

	biplot(pca, scaled_data, title=f"Biplot ({year}, {explanation})", color=scaled_data.PartyID)

	destination = args.outfile
	if not destination.suffix == ".pdf":
		destination = destination.with_suffix(".pdf")
	log.info(f"Saving to {destination}")
	plt.savefig(destination, bbox_inches='tight')

if __name__ == '__main__':
	main()
