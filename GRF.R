rm(list=ls())

library(tidyr)
library(grf)
library(mlr)


# Load data


tau.forest <- causal_forest(X, Y, W)