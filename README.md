### Overview  
***  
scikit-learn: Machine Learning in Python  
***  
### Install  
***  
scikit-learn requires:  
- Python (>= 2.6 or >= 3.3)
- NumPy (>= 1.61)
- SciPy (>= 0.9)

If you already have a working installation of numpy and scipy, the easiest way to install scikit-learn is using `pip`  
`pip install -U scikit-learn`   
##-U means update  

or `conda`:  
`conda install scikit-learn`  
***  
### Usage  
***  
##example 1 - classifier  
```
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data ##All attributes of this flower are stored in `iris.data`
iris_Y = iris.target ##All classifications of this flower are stored in `iris.target`

print(iris_X[:2,:])
print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size = 0.3) ##The proportion of the test set accounts for 30%.

knn = KNeighborsClassifier() ##Define which module to use in scikit-learn.
knn.fit(X_train, y_train) ##training step

print(knn.predict(X_test)) ##Use the trained KNN model to predict values based on attributes.
print(y_test) ##Compared with true values
```
