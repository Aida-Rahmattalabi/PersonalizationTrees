import pandas as pd
from scipy.stats import bernoulli
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import random 
import numpy as np
import time
import math

import util


#---------------------------------------------------------------------------------------------------
def readData(address, file_name):
	pass
#---------------------------------------------------------------------------------------------------
def syntheticObservationalData(d, m, n, logPolicy): 
	# this function generates synthetic data as described in the paper "optimal rescriptive trees"
	# it takes as input d: dimention of the covariate vector, m: the number of treatments, ...
	# n: the sample size, logPolicy is the offline policy that generated the observational data
	
	X = np.array([[0.0 for i in range(d)] for j in range(n)], dtype = object)
	for i in range(n):
		for j in range(d):
			if(j%2 == 0):
				X[i,j] = norm.rvs(0, 1)
			else: 
				X[i,j] = bernoulli.rvs(0.5)

	#print('X: ', X)

	# define functions used for the outcome variable 
	def baseline0(x):
		return x[0] - x[1]**2# + x[2]

	def effect0(x):
		return -abs(x[3])# + x[2]

	Y = np.array([[0.0 for i in range(m)] for j in range(n)], dtype = object)

	for j in range(n):
		Y[j][0] = baseline0(X[j])
		Y[j][1] = effect0(X[j])

	#print('Y: ', Y)

	T = np.array([[0.0 for i in range(n)]], dtype = object)
	for i in range(n):
		T[0][i] = bernoulli.rvs(math.exp(Y[i][0])/(1 + math.exp(Y[i][0])))

	#print('T: ', T)

	YObs = np.array([[0 for i in range(n)]], dtype = object)

	for i in range(n):
		YObs[0][i] = Y[i][T[0][i]]

	#print('Observed Y: ', YObs)
	S = np.concatenate((np.concatenate((X, YObs.transpose()), axis = 1), T.transpose()), axis = 1)
	SFull = np.concatenate((np.concatenate((X, Y), axis = 1), T.transpose()), axis = 1)

	# split train and test data
	r = np.random.rand(S.shape[0])

	Train = S#[r<1, :]
	Test = SFull#[r>=0.5, :]

	return [Train, Test]
#---------------------------------------------------------------------------------------------------	
def printInorder(root): 
    if root != None: 
        # First recur on left child 
        printInorder(root.childLeft) 
  
        # then print the data of node 
        print('(feature, value): ', root.theta, root.j, root.data)
  
        # now recur on right child 
        printInorder(root.childRight)
#---------------------------------------------------------------------------------------------------	
def baseline(Train, Test, methods):
	T = set(Train[:,-1])
	m = len(T)

	value = [0 for a in methods]
	for a in range(len(methods)):
		if(methods[a] == 'RC-RF'): # regression and compare
			# random forest
			RCModels = []
			for i in range(m):
				X = Train[Train[:,-1] == i, 0:len(Train[0])-2]
				Y = Train[Train[:,-1] == i, -2]

				RC = RandomForestRegressor()
				RC.fit(X, Y)

				RCModels.append(RC)

			for i in range(len(Test)):
				yhat = [0 for i in range(m)]

				for j in range(m):
					RC = RCModels[j]

					x = Test[i, 0:len(Train[0])-2]
					y = Test[i,-2]
					yhat[j] = RC.predict(x.reshape(1,-1))

				p = np.argmin([yhat])
				value[a] += Test[i, len(x) + p]/len(Test)
		
		if(methods[a] == 'RC-LinReg'): # regression and compare
			# random forest
			RCModels = []
			for i in range(m):
				X = Train[Train[:,-1] == i, 0:len(Train[0])-2]
				Y = Train[Train[:,-1] == i, -2]

				RC = LinearRegression()
				RC.fit(X, Y)

				RCModels.append(RC)

			for i in range(len(Test)):
				yhat = [0 for i in range(m)]

				for j in range(m):
					RC = RCModels[j]

					x = Test[i, 0:len(Train[0])-2]
					y = Test[i,-2]
					yhat[j] = RC.predict(x.reshape(1,-1))

				p = np.argmin([yhat])
				value[a] += Test[i, len(x) + p]/len(Test)


	return value
#---------------------------------------------------------------------------------------------------	
def main_synthetic():
	d = 4
	n = 500
	m = 2
	doMatching = False
	treeNum = 50
	res = syntheticObservationalData(d, m, n, [])
	
	Train = res[0]
	Test = res[1]

	#Train = np.array([[1.72,-1, 1], [1.39, 0, 0], [-1.02, 0.0,  0], [0.18, 0.0 , 0]], dtype = object)
	#Test = np.array([[1.72 ,0 ,-1, 1], [1.39, 0, -1, 0], [-1.02,0 ,-1 ,0], [0.18, 0 ,-1, 0]], dtype = object)

	max_depth = 2
	min_leaf_number = 0
	
	# create the personalization tree
	pt = util.personalizationTree(Train, Test, max_depth, min_leaf_number, doMatching)

	#print('the personalization tree is: ')
	#printInorder(pt.root)

	value = pt.policyEvaluation()
	print('Personalization Tree value: ', value)

	pf = util.personalizationForest(Train, Test, max_depth, min_leaf_number, treeNum, doMatching)
	value = pf.policyEvaluation()
	print('Personalization Forest value: ', value)

	value_base = baseline(Train, Test, ['RC-RF', 'RC-LinReg'])
	print('Random Forest value: ', value_base)
	
#---------------------------------------------------------------------------------------------------	
if __name__ == "__main__":
	for i in range(20):
		print('trial #', i)
		e = main_synthetic()
		#if(e == 0):
		#	print('PROGRAM ENDED')
		#	break


