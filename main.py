import pandas as pd
from scipy.stats import multinomial
import random 


#-------------------------------------------------------------------------------	
def generate_data_guassian(d, n, m):
	from sklearn.datasets import make_gaussian_quantiles
	# d is the number of covariates 
	# n in the size of the dataset -- the number of data points
	# c is the number of classes
	X, Y = make_gaussian_quantiles(cov=4.,
								n_samples=n, n_features=d,
								n_classes=m, random_state=1)


	X = pd.DataFrame(X,columns=['x'+str(i) for i in range(d)])
	Y0 = pd.Series(Y)

	# print("X:", X)
	# print("Labels:", Y)
	
	return X, Y0
#-------------------------------------------------------------------------------	
def generate_complex_data(d, n, m):
	from sklearn.datasets import make_classification
	# d is the number of covariates 
	# n in the size of the dataset -- the number of data points
	# c is the number of classes
	X,Y = make_classification(n_samples=n, n_features=d, n_informative=2, 
								n_redundant=0, n_repeated=0, n_classes=m, 
								n_clusters_per_class=1,class_sep=2,flip_y=0.2, 
								weights=[0.5,0.5], random_state=17)


	X = pd.DataFrame(X,columns=['x'+str(i) for i in range(d)])
	Y0 = pd.Series(Y)

	# print("X:", X)
	# print("Labels:", Y)
	
	return [X, Y0]
#-------------------------------------------------------------------------------	
def read_data_(address, file_name):
	pass
#-------------------------------------------------------------------------------	
def create_observational_data(X,Y0):
	Y = Y0
	return [X,Y,Y0]
#-------------------------------------------------------------------------------	






if __name__ == "__main__":
	[X,Y0] = generate_data_guassian(4, 10, 3)
	[X,Y,Y0] = create_observational_data(X, Y0)

	


