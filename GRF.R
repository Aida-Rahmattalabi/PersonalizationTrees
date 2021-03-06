rm(list=ls())

library(tidyr)
library(grf)
library(ggplot2)
library("DiagrammeRsvg")
library("DiagrammeR")

n = 1000
d = 10
m = 2
splitVal = 0.5
synthetic = TRUE
finalVal <- c()
#########################  generate data #########################  
for(n in c(400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)) #400, 600, 800, 1000, 1200, 1400, 1600, 1800, 
{ 
  val = c()
  for(run in seq(0, 20, 1))
  {
    if(synthetic)
    {  
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
      
      names(X)[1:d] <- paste("x", 0:(d-1), sep="")
      r = runif(n, min = 0, max = 1)
      XTrain = X[r >= splitVal,]
      XTest = X[r < splitVal,]
      
      # Y
      Y = data.frame(matrix(nrow=n, ncol= m))
      names(Y)[1:m] <- paste("y", 0:(m-1), sep="")
      
      baseline1 <- function(X) 
      {
        tmp = data.frame(matrix(nrow=nrow(X), ncol= 1))
        tmp = X['x0'] + X['x2'] + X['x4'] + X['x6'] + X['x7'] + X['x8'] - 2
        return(tmp)
      }
      
      baseline2 <- function(X) 
      {
        return(0)
      }
      
      
      effect1 <- function(X) 
      {
        tmp = data.frame(matrix(nrow=nrow(X), ncol= 1))
        index = with(X, X['x0'] > 0.5)
        tmp[index] = 5
        tmp[!index] = -5
        
        return(tmp)
      }
      
      effect2 <- function(X) 
      {
        tmp = data.frame(matrix(nrow=nrow(X), ncol= 1))
        
        index = with(X, X['x0'] > 1 && X['x2'] > 0 && X['x4'] > 1 && X['x6'] > 0)  
        a = 8 + 2*X['x7']*X['x8']
        
        tmp[index] = a[index]
      
        index = with(X, ((X['x0'] <= 1 || X['x2'] <= 0) && (X['x4'] <= 1 || X['x6'] <= 0)))
        a = 2*X['x7']*X['x8']
        
        tmp[index] = a[index]
        
        index = with(X, ((X['x0'] <= 1 || X['x2'] <= 0) && (X['x4'] > 1 || X['x6'] > 0)) || ((X['x0'] > 1 || X['x2'] > 0) && (X['x4'] <= 1 || X['x6'] <= 0)) )
        a = 4 + 2*X['x7']*X['x8']
        
        tmp[index] = a[index]
        
        return(tmp)
      }
      
      for(i in seq(0, m-1))
      {
        if(i == 0)
        { Y['y0'] = baseline2(X) + 0.5*effect2(X)}#apply(X, 1, baseline0) + 0.5*apply(X, 1, effect0)}
        else
        { Y['y1'] = baseline2(X) - 0.5*effect2(X)}#apply(X, 1, baseline0) - 0.5*apply(X, 1, effect0)}
      }
      YTrain = Y[r >= splitVal,]
      YTest = Y[r < splitVal,]
      
    
      
      # T
      p = c(exp(Y['y0'])/(1+exp(Y['y0'])))
      W = matrix(rbinom(n, 1, p$y0), nrow=n, ncol=1)
      WTrain = W[r >= splitVal,]
      WTest = W[r < splitVal,]
      
    }
    ######################## random split #######################
    YObs <- matrix(NA, nrow = n, ncol = 1)
    YObs[W==0] = Y[W==0,'y0']
    YObs[W==1] = Y[W==1,'y1']
    
    YObsTrain = YObs[r >= splitVal,]
    YObsTest = YObs[r < splitVal,]
    ################# learn the causal forest #####################
    tau.forest <- causal_forest(as.matrix(XTrain), as.vector(YObsTrain), as.vector(WTrain), num.trees = 1)
    ########################### predict ############################
    c.pred <- predict(tau.forest, XTest)
    plot(tree <- get_tree(tau.forest, 1))
    ######################### evaluate policy ######################
    pValue = matrix(NA, nrow = nrow(XTest), ncol = 1)
    pValue[c.pred >= 0] = YTest[c.pred >= 0,'y0']
    pValue[c.pred < 0] = YTest[c.pred < 0,'y1']
    val <- c(val, mean(pValue))
    ################################################################   
  }
  finalVal <- rbind(finalVal, c(as.numeric(n), as.numeric(mean(val)), as.numeric(sd(val)), 'CF'))
}

plot.data = data.frame(finalVal)
names(plot.data) <- c('n','value','std','method')

data <- read.csv('/Users/aida/Dropbox/PhD/Courses/Data-Driven Optimization-Vishal/Project/PersonalizationTrees/results.csv')
plot.data <- rbind.data.frame(data, plot.data)

plot.data[,'n'] <- as.numeric(plot.data[,'n'])
plot.data[,'value'] <- as.numeric(plot.data[,'value'])
plot.data[,'std'] <- as.numeric(plot.data[,'std'])

plot.data[,'method'] <- factor(plot.data[,'method'],
                               levels = c("persTree", "persForest", "linReg","RF","CF"))
########################### plot the result ##############################   
ggplot(data=plot.data, aes(x=n, y=value, group = method)) + geom_line(aes(color= method),size=1.2) + geom_point(size = 4) +
  #geom_errorbar(aes(ymin=value - std, ymax = value + std), width=1) + 
  theme_minimal() + 
  theme(axis.text=element_text(size=22), axis.title = element_text(size=22), legend.text = element_text(size = 22), legend.title = element_text(size = 22)) +
  theme()
ggsave('/Users/aida/Dropbox/PhD/Courses/Data-Driven Optimization-Vishal/Project/Figures/perfdataNum.pdf', width = 8, height = 5)
##############################################################################
##############################################################################
##############################################################################
tmp <- read.csv('/Users/aida/Dropbox/PhD/Courses/Data-Driven Optimization-Vishal/Project/PersonalizationTrees/resultsbyTypex7baseline2.csv')
names(tmp) <- c("n", "group 1", "group 2", "std1", "std2", "method")
tmp[,'method'] <- factor(tmp[,'method'], levels = c("persTree", "persForest", "linReg","RF","CF"))

tmp2 <- rbind(tmp[, 1:5], c(2000, -2.0234600001, -1.16404300001	, 0.1502915900001, 0.123046200001))
tmp3 <- as.factor(c("persTree", "persForest", "linReg", "RF", "CF"))
tmp4 <- cbind(tmp2, tmp3)
names(tmp4)[6] <- 'method'
plot.data.fair <- gather(tmp4, condition, measurement, c('group 1', 'group 2'), factor_key=TRUE)


names(plot.data.fair)[6] <- "value"

ggplot(plot.data.fair, aes(fill=condition, y=value, x=method)) + 
  geom_bar(position="dodge", stat="identity") +
  theme_minimal() + 
  theme(axis.title.x = element_blank()) +
  theme(axis.text=element_text(size=22), axis.title = element_text(size=22), legend.text = element_text(size = 22), legend.title = element_blank())

ggsave('/Users/aida/Dropbox/PhD/Courses/Data-Driven Optimization-Vishal/Project/Figures/fairx7.pdf', width = 10, height = 5)


