rm(list=ls())

library(tidyr)
library(grf)
library(mlr)

n = 10
d = 3
m = 2

############# generate data #############  

# X
X = data.frame(matrix(nrow=n, ncol=0))

for(i in seq(0, d-1))
{
  if(i%%2 == 0)
  { tmp = data.frame(matrix(rnorm(n, 0, 1), nrow=n)) }
  else
  { tmp = data.frame(matrix(rbinom(n, 1, 0.5), nrow=n)) }

  X = cbind(X, tmp)
}
names(X)[1:d] <- paste("x", 1:d, sep="")


# Y
Y = data.frame(matrix(nrow=n, ncol=0))




for(i in seq(0, m-1))
{
  if(i%%2 == 0)
  { tmp = data.frame(matrix(rnorm(n, 0, 1), nrow=n)) }
  else
  { tmp = data.frame(matrix(rbinom(n, 1, 0.5), nrow=n)) }
  
  Y = cbind(Y, tmp)
}
names(Y)[1:d] <- paste("x", 1:d, sep="")


# T
W = data.frame(matrix(nrow=n, ncol=0))

for(i in seq(0, m-1))
{
  if(i%%2 == 0)
  { tmp = data.frame(matrix(rnorm(n, 0, 1), nrow=n)) }
  else
  { tmp = data.frame(matrix(rbinom(n, 1, 0.5), nrow=n)) }
  
  W = cbind(W, tmp)
}
names(W)[1:d] <- paste("x", 1:d, sep="")

################## random split #################

########### learn the causal forest #############
tau.forest <- causal_forest(X, Y, W)
############ ############ ############ ##########



