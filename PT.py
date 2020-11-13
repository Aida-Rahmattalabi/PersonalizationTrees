# This is an implementation of Algorithm 1 in the paper 
# "Recursive Partitioning for Personalization using Observational Data"

import numpy as np


class node():
	def __init__(self, l, v):
		self.l = l
		self.v = v

		self.childLeft = None
		self.childRight = None




class personalizationTree():

	def __init__(self, X, Y, T, max_depth, min_leaf_number):
		self.X = X
		self.Y = Y
		self.Z = Z
		self.max_depth = max_depth
		self.min_leaf_number = min_leaf_number

		self.d = len(X[0])  				# number of features
		self.m = 0
		self.candidateCuts = sqrt(self.d)

		
		self.subroutine(0)


	def dataProcessing():
		pass
		# implement later for data processing


	def subroutine(cur_depth)
		for l in range(d):
		


