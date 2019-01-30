"""Plot stackplot of explained variance."""
import argparse
import logging
import sys


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functools import reduce
from itertools import zip_longest
from pathlib import Path

parent_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder)) # Adds higher directory to python modules path.

from features.build_features import make_pca_and_scaled_data
from visualize import extract_year

log = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO)

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument("-i", "--infile", type=Path, nargs='+')  # At least one file.
	parser.add_argument("-o", "--outfile", type=Path, required=True)
	parser.add_argument("-s", "--strategy", type=str, choices={"impute", "drop"}, default="drop")
	return parser.parse_args()

def main():
	args = parse_args()
	datafiles = sorted(args.infile)
	log.info(f"Reading: {[str(x) for x in datafiles]}")

	years = [extract_year(file) for file in datafiles]
	data_frames = [pd.read_csv(datafile, index_col=0) for datafile in datafiles]
	for year, datafile in zip(years, datafiles):
		log.info(f"Year {year}: {datafile}")

	# Get variables consistent across all years.
	all_variables_list = [set(df.columns) for df in data_frames]
	shared_variables = list(reduce(set.intersection, all_variables_list))
	log.info(f"Shared variables: {', '.join(shared_variables)}")

	datas_and_pcas = [make_pca_and_scaled_data(df, missing_strategy=args.strategy) for df in data_frames]
	pcas = [pca for data_scaled, pca in datas_and_pcas]
	evrs = [pca.explained_variance_ratio_ for pca in pcas]

	plt.stackplot(years, list(zip_longest(*evrs, fillvalue=0)), labels=list(range(len(evrs[0]))))
	plt.legend()

	destination = args.outfile
	if not destination.suffix == ".pdf":
		destination = destination.with_suffix(".pdf")
	log.info(f"Saving to {destination}")
	plt.savefig(destination, bbox_inches='tight')

if __name__ == '__main__':
	main()
