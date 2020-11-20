# Author: Aida Rahmattalabi

# This is an implementation of Algorithm 1 in the paper 
# "Recursive Partitioning for Personalization using Observational Data"

import numpy as np
from numpy.linalg import inv
from scipy.spatial import distance
import pandas as pd
import random 
import math
#---------------------------------------------------------------------------------------------------	
class Node():
	def __init__(self, S, j, theta): # each node is characterized by a feature (index) and split value.
		self.j = j
		self.theta = theta

		self.data = S
		self.childLeft = None
		self.childRight = None
#---------------------------------------------------------------------------------------------------	
class personalizationTree():
	def __init__(self, Train, Test, max_depth, min_leaf_number, doMatching):
		
		self.Train = Train
		self.Test = Test
		self.max_depth = max_depth
		self.min_leaf_number = min_leaf_number

		self.d = int(len(Train[0]) - 2) 					# number of features
		T = Train[0:len(Train),-1] 
		self.m = len(set(T))							# number of treatments

		self.candidateCuts = self.d#int(math.sqrt(self.d))
		
		if(doMatching):
			self.nTest = math.floor(0.2*len(Train))
			res = self.subsetMatching(S)

			self.Train = res[0]
			self.Test = res[1]
			self.root = Node(self.Train, 0,0)

			self.subroutine(self.Train, 0, self.root)
		else:
			self.root = Node(self.Train, 0,0)

			self.subroutine(self.Train, 0, self.root)
	#-----------------------------------------------------------------------------------------------
	def treeTraversal(self, node, x):
		if(node.childLeft == None or node.childRight == None):
			res = []

			T = node.data[:,-1]
			Y = node.data[:,-2]

			for t in range(self.m):
				ind = [i for i, x in enumerate(T) if x == t]
				if(len(ind) != 0):
					tmp = [Y[i]/len(ind) for i in ind]

					res.append(sum(tmp))
				else:
					res.append(math.inf)

			return np.argmin(res)
		else:
			j = node.j
			theta = node.theta
			
			if(x[j] <= theta):
				return self.treeTraversal(node.childLeft, x)
			else:
				return self.treeTraversal(node.childRight, x)
	#-----------------------------------------------------------------------------------------------
	def policy(self, x):
		return self.treeTraversal(self.root, x)
	
	#-----------------------------------------------------------------------------------------------
	def subroutine(self, S, cur_depth, node):
		# count the number of data points for each treatment bucket
		res = []
		T = S[0:len(S), -1]

		for t in range(self.m):
			ind = [i for (i, x) in enumerate(T) if x == t]
			res.append(len(ind))

		if(cur_depth < self.max_depth and min(res) > self.min_leaf_number):
			IStar = math.inf													# line 4
			lStar = 0															# line 4
			jStar = 0															# line 4
			cuts = list(np.random.randint(0, self.d, self.candidateCuts))		# line 5
			for l in cuts:  # feature 
				# sort X along l-th feature 
				S = sorted(S, key=lambda axis:axis[l], reverse=False)
				S = np.array(S)
				
				T = S[0:len(S),-1]
				Y = S[0:len(S),len(S[0])-2]
				X = S[0:len(S),0:len(S[0])-2]

				k_mL = [0 for i in range(self.m)]		# left nodes 
				S_mL = [0 for i in range(self.m)]		# left nodes
				k_L = 0									# left nodes

				k_mR = [len([i for (i,x) in enumerate(T) if x == t]) for t in range(self.m)]
				S_mR = []

				for t in range(self.m):
					ind = [i for i, x in enumerate(T) if x == t]
					tmp = sum([Y[i] for i in ind])
					S_mR.append(tmp)

				k_R = int(len(X))

				for j in range(len(X)-1): # the cut index
					k_L += 1
					k_R -= 1
					t = T[j]

					k_mL[t] += 1
					k_mR[t] -= 1

					S_mL[t] += Y[t]
					S_mR[t] -= Y[t]

					if(0 in k_mL and 0 not in k_mR):
						I = k_R*min([S_mR[i]/k_mR[i] for i in range(self.m)])

					elif(0 in k_mR and 0 not in k_mL):
						I = k_L*min([S_mL[i]/k_mL[i] for i in range(self.m)])
					
					elif(0 in k_mR and 0 in k_mL):
						I = IStar

					else:
						I = k_L*min([S_mL[i]/k_mL[i] for i in range(self.m)]) + k_R*min([S_mR[i]/k_mR[i] for i in range(self.m)])
					
					k_min = min(min(k_mR), min(k_mL))

					if I < IStar and k_min >= self.min_leaf_number:
						IStar = I
						lStar = l
						jStar = j


			if IStar < math.inf:
				S = sorted(S, key=lambda axis:axis[lStar], reverse=False)
				S = np.array(S)

				S_L = S[0:jStar+1, 0:len(S[0])]
				S_R = S[jStar+1:len(S), 0:len(S[0])]

				leftNode = Node(S_L, 0,0)
				node.childLeft = leftNode

				rightNode = Node(S_R, 0,0)
				node.childRight = rightNode

				self.subroutine(S_L, cur_depth + 1, leftNode)
				self.subroutine(S_R, cur_depth + 1, rightNode)

				node.j = lStar
				node.theta =  0.5*(X[jStar,lStar] + X[jStar+1,lStar]) 
	#-----------------------------------------------------------------------------------------------
	def distanceFunction(self):
		# this function implements Mahalanobis distance between all data points in S. 
		# it is used withing the sub-matching 
		d = [[0 for i in range(len(S))] for j in range(len(S))]
		X = S[0:len(S), 0:len(S[0])-2]
		Sigma = np.cov(S, rowvar=False)

		return d
	#-----------------------------------------------------------------------------------------------
	def subsetMatching(self, S):
		# this function takes as input a function, i.e., policy and a dataset 
		# to use for policy evaluation. For policy evaluation, we use sub-matching
		X = S[0:len(S), 0:len(S[0])-2]
		T = S[0:len(S), -1]
		Y = S[0:len(S), -2]
		
		# extract a test set
		testSet = list(np.random.randint(0, len(S), self.nTest))	
		XTest = X[testSet, :]
		YTest = [[0 for j in range(self.m)] for i in range(self.nTest)]
		STest = np.concatenate((XTest, YTest), axis = 1)
		flag = []  # used to flag the points used for evaluation

		# evaluate the inverse of the covariance matrix
		Sigma = np.cov(X.astype(float), rowvar=False)
		Sigmainv = inv(Sigma)

		d = [[0 for j in range(len(X))] for i in range(len(XTest))]
		for i in range(len(XTest)):
			for j in range(len(X)):
				d[i][j] = math.sqrt(np.matmul(np.matmul(XTest[i]-X[j], Sigmainv),
				 (XTest[i] - X[j]).transpose()))


		for j in range(self.nTest):
			flag.append(testSet[j])

			for m in range(self.m):

				x = S[testSet[j], 0:len(S[0])-2]
				t = S[testSet[j], -1]
				y = S[testSet[j], -2]	

				if(m == t):
					STest[j][self.d+t] = y
				else:
					sameT = [i for i, x in enumerate(d[j]) if T[i] == m]
					i = np.argmin(sameT)
					y = S[i, -2]	
					np.append(flag,i)

					Test[j][self.d+m] = y

		Train = np.delete(S, flag, axis = 0)

		return [Train, Test]
	#-----------------------------------------------------------------------------------------------
	def policyEvaluation(self):
		# policy value
		val = 0
		for i in range(len(self.Test)):

			# find the covariate vector
			x = self.Test[i,0:self.d]
			
			# find the action proposed by policy f
			p = self.treeTraversal(self.root, x)

			# evaluate the action
			val += self.Test[i,self.d + p]

		return val/len(self.Test)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class personalizationForest():
	def __init__(self, Train, Test, max_depth, min_leaf_number, treeNum, doMatching):		
		self.Train = Train
		self.Test = Test
		self.max_depth = max_depth
		self.min_leaf_number = min_leaf_number
		self.treeNum = treeNum
		self.doMatching = doMatching

	def policyEvaluation(self):
		value = 0
		treeModels = []

		for i in range(self.treeNum):
			# randomly sample training data
			Train_s = self.Train[np.random.choice(self.Train.shape[0], len(self.Train), replace=True), :]
			tree = personalizationTree(Train_s, self.Test, self.max_depth, self.min_leaf_number, self.doMatching)

			treeModels.append(tree)

		for j in range(len(self.Test)):
			zhat = [0 for i in range(self.treeNum)]

			for i in range(self.treeNum):
				tree = treeModels[i]

				x = self.Test[j, 0:len(self.Train[0])-2]
				zhat[i] = tree.policy(x)

			p = max(set(zhat), key=zhat.count)
			value += self.Test[j, len(x) + p]/len(self.Test)



		return value

