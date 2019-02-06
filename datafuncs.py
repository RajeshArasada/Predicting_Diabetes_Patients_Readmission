
import pandas as pd


def data_missingness(dataframe):
	assert isinstance(dataframe, pd.DataFrame)
	num_missing_values = dataframe.isnull().sum()
	percent_missing_values = 100 * num_missing_values/dataframe.shape[0]

	# make a dataframe
	missing_data_df = pd.DataFrame({'# missing values': num_missing_values,
									'Percent of missing values': percent_missing_values})
	return missing_data_df


def subset_dataframe(dataframe, dtype_):
	"""
	Pass 'object as a second argument if a dataframe with categorical features is needed.
	Pass np.number or 'number' if a dataframe with numerical features is required.
	"""
	assert isinstance(dataframe, pd.DataFrame)
	return dataframe.select_dtypes(include=dtype_)



def unique_values_categorical_features(dataframe):
	
	assert isinstance(dataframe, pd.DataFrame)
	df = dataframe.select_dtypes(include=object)
	unique_dict = {}
	for col in df.columns.tolist():
		assert isinstance(dataframe[col], object)
		unique_dict[col] = df[col].unique()
	return unique_dict


def value_counts_categorical_features(dataframe):
	assert isinstance(dataframe, pd.DataFrame)
	df = dataframe.select_dtypes(include=object)
	value_counts_dict = {}
	for col in df.columns.tolist():
		assert isinstance(dataframe[col], object)
		value_counts_dict[col] = df[col].value_counts()
	return value_counts_dict

