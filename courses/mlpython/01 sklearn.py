# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:43:06 2018

@author: jpmaldonado
"""


'''
LECTURE OUTLINE

* What is machine learning? (Discussion, examples)
* Goals of machine learning:
        - Automatic decision making
        - Generalization

* Supervised Learning
    - Classification
    - Regression
    - Ranking
* Unsupervised Learning
    - Clustering
    - Dimensionality Reduction
    - Anomaly Detection
    
* scikit-learn
'''

####################################
## Data representation in sklearn
####################################

# In scikit-learn data is represented as a two dimensional array
# of shape [n_samples, n_features]

# Tabular data
from sklearn.datasets import load_iris

iris = load_iris() # a 'Bunch' object, some sort of dict

iris.keys() # See what's inside

n_samples, n_features = iris.data.shape
print(iris.target)
print(iris.target.shape)
print(iris.data.shape)
print(iris.target_names)

# Image data
from sklearn.datasets import load_digits
digits = load_digits()

# What's inside?
digits.keys()
digits.images[0]
digits.data[0]

import matplotlib.pyplot as plt

plt.imshow(digits.images[0])
plt.imshow(digits.images[0], cmap = plt.cm.binary)

####################################
# Classification
####################################

# Whic model to use?
# http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

X, y = iris.data, iris.target

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

# Train/ test split
y[0:50]
y[50:100]
y[100:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8,
                                                    test_size = 0.2, 
                                                    random_state = 123, 
                                                    stratify = y)

clf.fit(X_train,y_train) # Fit the model
y_preds = clf.predict(X_test) # Prediction
sum(y_test!=y_preds) # How many errors?
y_preds[y_test != y_preds] #Where is the error?
y_test[y_test != y_preds]
## QUESTION: Does this suggest a way to improve our classifier?

#Try a different classifier: logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
sum(y_test != y_preds)

# LR can also predict class probabilities
clf.predict_proba(X_test)


####################################
# Regression
####################################
import numpy as np

x = np.linspace(-3,3,100)
y = 2*x + 3 + 0.1*np.random.normal(size=len(x))

plt.scatter(x,y)


x = x.reshape(-1,1)

from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size = 0.8, 
                                                    test_size = 0.2, 
                                                    random_state = 123
                                                    )

clf = LinearRegression()
clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)


# Measuring error: MSE

y_preds = clf.predict(x_test)
MSE = sum((y_preds-y_test)**2)/len(y_test)

clf.predict([[4.5]])

## EXERCISE: Try linear regression on the Boston dataset
from sklearn.datasets import load_boston
boston = load_boston()