import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def drop_missing(X):
	"""Delete rows (respondents) with any missing values."""
	return X.dropna(axis='index', how='any')

def make_pca_and_scaled_data(data: pd.DataFrame, *, missing_strategy: str):

	# Make a handler for missing data: fill with mean or drop.
	if missing_strategy == "impute":
		missing_handler = SimpleImputer(strategy='mean')
	elif missing_strategy == "drop":
		missing_handler = FunctionTransformer(drop_missing, validate=False)
	else:
		raise ValueError(f"missing_strategy '{missing_strategy}' is not 'drop' or 'impute'")

	scaler = StandardScaler()
	pca = PCA()

	preprocessing = Pipeline([  # Just the non-PCA components.
		('mis', missing_handler),
		('scl', scaler),
	])
	pipeline = Pipeline([
		('pre', preprocessing),
		('pca', pca),
	])

	pipeline.fit(data)  # Fit parameters for all components.
	_scaled = preprocessing.transform(data)
	data_scaled = pd.DataFrame(_scaled, columns=data.columns)

	# Return data and fitted PCA object.
	return data_scaled, pca

