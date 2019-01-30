"""Plot Spearman's correlation matrix."""
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

log = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO)

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument("-i", "--infile", type=Path, required=True)
	parser.add_argument("-o", "--outfile", type=Path, required=True)
	return parser.parse_args()

def main():
	args = parse_args()
	datafile = args.infile
	log.info(f"Reading {datafile}")

	df = pd.read_csv(datafile, index_col=0)

	log.info("Creating Spearman's correlations plot ")
	correlations = df.corr(method='spearman')
	plt.figure()
	sns.heatmap(correlations, square=True)

	destination = args.outfile
	if not destination.suffix == ".pdf":
		destination = destination.with_suffix(".pdf")
	log.info(f"Saving to {destination}")
	plt.savefig(destination, bbox_inches='tight')

if __name__ == '__main__':
	main()