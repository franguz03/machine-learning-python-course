Questions to understand the data

## supervised learning
It give us a direct route to transform data in real an processable information.
Help us to prevent unwanted results.
Molds must be rebuilt periodically.


## No supervised

No labeled data: Unsupervised learning works with unlabeled data, meaning the algorithm doesn't have predefined outputs.

Pattern discovery: The goal is to find hidden patterns or structures in the data, like clustering or associations.

Common techniques: Includes clustering (e.g., K-means, hierarchical clustering) and dimensionality reduction (e.g., PCA).

Applications: Used in anomaly detection, market segmentation, and recommendation systems.

## Clasification vs Regression

predict a class vs predict a number value

## tranform data

fit() to learn how to transform
transform() to transform the dataset

## Transformation algorithms

### escalation
MinMAxScaler : for gradient descent (optimization algorithms), regression, neural networks, k-nearest
### standardization
StandarScaler(): mean value of zero and standard deviation of 1
### binarization
Binarizer(threshold=):with a threshold you difine if a value wil be  0 or 1
### box-cox
PowerTransformer(method='box-cox',standardize=True): If the information does not have a gaussian distribution,this transformation gives.
we must have only positive value in the feature, only for features with bias
### yeo-johnson
PowerTransformer(method='yeo-johnson',standardize=True);like box-cox but this one accepts  negative values
