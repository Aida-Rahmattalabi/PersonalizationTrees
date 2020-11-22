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
from statistics import stdev 
from statistics import mean 


import util
#---------------------------------------------------------------------------------------------------	
def readData(address, file_name):
	pass
#---------------------------------------------------------------------------------------------------
def syntheticObservationalData(d, m, n): 
	# this function generates synthetic data as described in the paper "optimal rescriptive trees"
	# it takes as input d: dimention of the covariate vector, m: the number of treatments, ...
	# n: the sample size, logPolicy is the offline policy that generated the observational data
	
	# np.random.seed(1)
	X = np.array([[0.0 for i in range(d)] for j in range(n)], dtype = object)
	for i in range(n):
		for j in range(d):
			if(j%2 == 0):
				X[i,j] = norm.rvs(0, 1)
			else: 
				X[i,j] = bernoulli.rvs(0.5)

	#print('X: ', X)

	# define functions used for the outcome variable 
	def baseline1(x):
		return x[0] + x[2] + x[4] + x[6] + x[7] + x[8] - 2

	def effect1(x):
		if(x[0] > 0.5):
			return 5
		else:
			return -5

	def baseline2(x):
		return 0

	def effect2(x):
		if(x[0] > 1 and x[2] > 0 and x[4] > 1 and x[6] > 0):
			
			return 8 + 2*x[7]*x[8]
		if((x[0] <= 1 or x[2] <= 0) and (x[4] <= 1 or x[6] <= 0)):

			return 2*x[7]*x[8]
		if(((x[0] <= 1 or x[2] <= 0) and (x[4] > 1 and x[6] > 0)) or ((x[0] > 1 and x[2] > 0) 
			and (x[4] <= 1 or x[6] <= 0))):

			return 4 + 2*x[7]*x[8]

	Y = np.array([[0.0 for i in range(m)] for j in range(n)], dtype = object)

	for j in range(n):
		Y[j][0] = baseline1(X[j]) + 0.5*effect1(X[j])
		Y[j][1] = baseline1(X[j]) - 0.5*effect1(X[j])

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

	Train = S[r<0.5, :]
	Test = SFull[r>=0.5, :]

	return [Train, Test]
#---------------------------------------------------------------------------------------------------	
def printInorder(root): 
    if root != None: 
        # First recur on left child 
        printInorder(root.childLeft) 
  
        # then print the data of node 
        print('(feature, value): ', root.theta, root.j)
  
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
					yhat[j] = RC.predict(x.reshape(1,-1))

				p = np.argmin([yhat])
				value[a] += Test[i, len(x) + p]/len(Test)

		if(methods[a] == 'CF'):
			pass
	return value
#---------------------------------------------------------------------------------------------------	
def main_synthetic(runs):

	baseAlgorithms = ['RC-RF', 'RC-LinReg']

	baseVals = [[0 for i in range(len(baseAlgorithms))] for j in range(runs)]
	persTreeVals = [0 for i in range(runs)]
	persForestVals = [0 for i in range(runs)]

	# save results to file
	f = open('results.csv', 'w')
	f.write('n, value, std, method \n')

	for n in [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]:#[400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]:
		for run in range(runs):
			d = 10
			#n = 1000
			m = 2
			doMatching = False
			treeNum = 10
			res = syntheticObservationalData(d, m, n)

			Train = res[0]
			Test = res[1]

			max_depth = 4
			min_leaf_number = 5
			# create the personalization tree
			pt = util.personalizationTree(Train, Test, max_depth, min_leaf_number, doMatching)

			print('the personalization tree is: ')
			printInorder(pt.root)

			value = pt.policyEvaluation()
			print('Personalization Tree value: ', value)
			persTreeVals[run] = value

			pf = util.personalizationForest(Train, Test, max_depth, min_leaf_number, treeNum, doMatching)
			value = pf.policyEvaluation()
			print('Personalization Forest value: ', value)
			persForestVals[run] = value


			value_base = baseline(Train, Test, baseAlgorithms)
			print('baseline values (RF, Linear Regression): ', value_base)
			for i in range(len(value_base)):
				baseVals[run][i] = value_base[i] 
		
		f.write(str(n)+','+str(mean(persTreeVals))+','+str(stdev(persTreeVals))+','+'persTree'+'\n')
		f.write(str(n)+','+str(mean(persForestVals))+','+str(stdev(persForestVals))+','+'persForest'+'\n')
		f.write(str(n)+','+str(mean(baseVals[0]))+','+str(stdev(baseVals[0]))+','+'RF'+'\n')
		f.write(str(n)+','+str(mean(baseVals[1]))+','+str(stdev(baseVals[1]))+','+'linReg'+'\n')

#---------------------------------------------------------------------------------------------------	
def main_real():
	d = 10
	n = 1000
	m = 2
	doMatching = True
	
	res = syntheticObservationalData(d, m, n)
	#S = readDate()

	max_depth = 4
	min_leaf_number = 5
	# create the personalization tree
	pt = tree.personalizationTree(res[0], res[1], max_depth, min_leaf_number, doMatching)

	#print('the personalization tree is: ')
	#printInorder(pt.root)

	value = pt.policyEvaluation(pt.policy)
	print('value: ', value)


	value_base = baseline(Train, Test, ['RC-RF', 'RC-LinReg'])
	print('baseline values (RF, Linear Regression): ', value_base)
	


#---------------------------------------------------------------------------------------------------	
if __name__ == "__main__":
	main_synthetic(25)


