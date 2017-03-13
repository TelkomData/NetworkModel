import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def y_location(df, y_cat='location_code_summary', y_labels=['CORE', 'MDF', 'E-SIDE', 'SDC', 'V-SIDE', 'LAST MILE', 'CUSTOMER']):
#predict where
	
	y_data = pd.pivot_table(df, 
	               index='master_ticket_id', 
	               columns=y_cat, 
	               values='dn_number', 
	               aggfunc='count')

	y_data = (y_data > 0).astype('int')
	df = df.join(y_data, on=['master_ticket_id'], how='left')
	df[y_labels] = df[y_labels].fillna(0, axis=1)
	return df[y_labels]

#predict network fault
def y_cause(df, y_cat='cause_code_summary', y_labels = 'Network Issues'):	
	df[y_labels] = (df[y_cat] == y_labels).astype('int')
	return df[[y_labels]]

def x_features(df, x_cat_cols=['sdc', 'dp']):
	x_cont_cols = list(df.columns[13:])
	df[x_cat_cols] = df[x_cat_cols].fillna('', axis=1)
	df[x_cont_cols] = df[x_cont_cols].fillna(0, axis=1)

	cat_labels = {}
	for cat in x_cat_cols:
	    cat_labels[cat] = LabelEncoder().fit(df[cat])
	    df[cat+'_index'] = cat_labels[cat].transform(df[cat])
	feature_cols = [x + '_index' for x in x_cat_cols] + x_cont_cols
	return df[feature_cols]

def x_sum_features(df, x_cat_cols=['sdc', 'dp']):
	x_cont_cols = list(df.columns[13:])
	df['open_faults'] = df[x_cont_cols].sum(axis=1)
	df['ave_sdc'] = df['open_faults']/(df[x_cont_cols]).shape[1]
	df['number_of_sdc_reporting'] = (df[x_cont_cols]>0).sum(axis=1)
	df['prop_of_sdc_reporting'] = df['number_of_sdc_reporting']/(df[x_cont_cols]).shape[1]
	df['ave_sdc_with_fault'] = df['open_faults']/df['number_of_sdc_reporting']
	x_cont_cols = list(df.columns[-5:])
	df[x_cat_cols] = df[x_cat_cols].fillna('', axis=1)

	df[x_cont_cols] = df[x_cont_cols].fillna(0, axis=1)
	cat_labels = {}
	for cat in x_cat_cols:
		cat_labels[cat] = LabelEncoder().fit(df[cat])
		df[cat+'_index'] = cat_labels[cat].transform(df[cat])
	feature_cols = [x + '_index' for x in x_cat_cols] + x_cont_cols
	return df[feature_cols]
	

def load_exchange(exc, x_sum=True, y_loc = True, train_size=0.8, random_state=42): 

	# get data in right format
	df = pd.read_csv(exc+'.csv')
	if x_sum:
		x_data = x_sum_features(df)
	else:
		x_data = x_features(df)

	if y_loc:
		y_data = y_location(df)
	else:
		y_data = y_cause(df)

	X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=42)

	return X_train, X_test, y_train, y_test

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = load_exchange('Exchanges/RPOA', y_loc=False)
	print (X_train.head())
	print (y_train.head())