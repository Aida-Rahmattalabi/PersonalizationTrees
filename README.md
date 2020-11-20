# PersonalizationTrees

This project reproduces the paper "Recursive Partitioning for Personalization using Observational Data". It implements the personalization trees (see Algorithm 1 in the paper) and personalization forest (see Algorithm 2). 

The repository consists of two files: 
<b> util.py </b> includes two classes: 1) personlizationTree and 2) personalizationForest. 

<b> personlizationTree </b> takes 5 inputs. 
<ol>
  <li>Train, datatype = numpy array (object): training set in the form of [X,Y,T] where X are the covariates, Y is the observed outcome and T is the assigned treatment.</li> 
  <li>Test</li>: set set in the form of [X,Y,T] where X are the covariates, Y is the observed outcome and T is the assigned treatment.
  <li>max_depth</li>
  <li>min_leaf_number</li>
  <li>doMatching</li>
</ol>
