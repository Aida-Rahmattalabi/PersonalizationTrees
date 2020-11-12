# This is an implementation of the decision tree of the paper 
# "Recursive Partitioning for Personalization using Observational Data"


def PT(Data, current_depth, max_depth, min_leaf_number, feature_num):
	X = Data[0]	  # covariates
	T = Data[1]	  # treatment
	Y = Data[2]   # outcome

	d = len(X[0])  # number of features
	m = max(Y) + 1 # number of actions/treatments

	for l in range(d):
		



