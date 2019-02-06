import datafuncs as dfn
import pytest
import numpy as np
import pandas as pd


def test_data_missingness():
	 # set up the test dataframe
	df = pd.DataFrame({'A': [5, 10, 5, 6, 8],							
						'B': ['Matt', 'Chad', 'Chen', 'Sofia', 'Moon']})
	 # collect the result into a variable
	df_missingness = dfn.data_missingness(df)
	for col in df_missingness:
		assert df_missingness[col].sum() == 0


def test_subset_dataframe():
	# set up the test dataframe
	df = pd.DataFrame({'A': [5, 10, 5, 6, 8],
						'B': ['Matt', 'Chad', 'Chen', 'Sofia', 'Moon'],
						'C': ['Yes', 'No', 'No', 'No', 'Yes']})
	# collect the result into a variable
	df_cat = dfn.subset_dataframe(df, object)
	assert df_cat.columns.tolist() == ['B','C']
	# collect the result into a variable
	df_numeric = dfn.subset_dataframe(df, 'number')
	assert df_numeric.columns.tolist() == ['A']


def test_unique_values_categorical_features():
	# set up the test dataframe
	df = pd.DataFrame({'A': [5, 10, 5, 6, 8],
						'B': ['Matt', 'Chad', 'Chen', 'Sofia', 'Moon'],
						'C': ['Yes', 'No', 'No', 'No', 'Yes']})
	# collect the result into a variable
	unique_values = dfn.unique_values_categorical_features(df)
	assert unique_values['B'].all() == np.array(df['B'].all())


def test_value_counts_categorical_features():
	# set up the test dataframe
	df = pd.DataFrame({'A': [5, 10, 5, 6, 8],
						'B': ['Matt', 'Chad', 'Chen', 'Sofia', 'Moon'],
						'C': ['Yes', 'No', 'No', 'No', 'Yes']})
	# collect the result into a variable
	counts = dfn.value_counts_categorical_features(df)
	assert counts['B'].all() == df['B'].value_counts().all()

