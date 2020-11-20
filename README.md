# PersonalizationTree/Forest

This project reproduces the paper "Recursive Partitioning for Personalization using Observational Data". It implements the personalization trees (see Algorithm 1 in the paper) and personalization forest (see Algorithm 2). 

The repository consists of two files: 
<b> util.py </b> includes two classes: 1) personlizationTree and 2) personalizationForest. 

<b> personlizationTree </b> takes 5 inputs. 
<ol>
  <li><b>Train (datatype = numpy array, object)</b>: training set in the form of [X,YObs,T] where X are the covariates, YObs is the observed outcome and T is the assigned treatment.</li> 
  <li><b>Test (datatype = numpy array, object)</b>: set set in the form of [X,Y,T] where X are the covariates, Y collects the potential outcome.</li>
  <li><b>max_depth (datatype = int)</b>: the maximum depth of the personalization tree. </li>
  <li><b>min_leaf_number (datatype = int)</b>: the minimum number of data points in each leaf node. </li>
  <li><b>doMatching (datatype = Boolean)</b>: a flag that, when set to True, signals the personalization tree to use the greedy-submatching (see Section 3 in the paper) to estimate the potential outcomes. Else, the user should provide the potential outcomes in Test. </li>
</ol>

<b> personlizationForest </b> takes <b>treeNum (datatype = int)</b> which is the number of trees in addition to the above inputs. 


