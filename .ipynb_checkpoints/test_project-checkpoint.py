# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:08:16 2020

@author: 偉庭
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv')
data1 = pd.read_csv('test_1.csv')
X = data.iloc[:, 0:34].values
y = data.iloc[:, 34].values.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
ONE = OneHotEncoder()
X1 = X[:, 9].reshape(-1, 1)
X1 = ONE.fit_transform(X1).toarray()

X2 = X[:, 10].reshape(-1, 1)
X2 = ONE.fit_transform(X2).toarray()

X3 = X[:, 11].reshape(-1, 1)
X3 = ONE.fit_transform(X3).toarray()

X4 = X[:, 8].reshape(-1, 1)
X4 = ONE.fit_transform(X4).toarray()

X5 = X[:, 7].reshape(-1, 1)
X5 = ONE.fit_transform(X5).toarray()

X6 = X[:, 6].reshape(-1, 1)
X6 = ONE.fit_transform(X6).toarray()

X = np.c_[X6, X5, X4, X1, X2, X3, X[:, [23, 24]], X[:, 26:32], X[:, 33]]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.032, C =0.69, random_state = 0)
classifier.fit(X_train, y_train)

Z = data1.iloc[:, :].values

Z1 = Z[:,9].reshape(-1,1)
Z1 = ONE.fit_transform(Z1).toarray()

Z2 = Z[:,10].reshape(-1,1)
Z2 = ONE.fit_transform(Z2).toarray()

Z3 = Z[:,11].reshape(-1,1)
Z3 = ONE.fit_transform(Z3).toarray()

Z4 = Z[:,8].reshape(-1,1)
Z4 = ONE.fit_transform(Z4).toarray()

Z5 = Z[:,7].reshape(-1,1)
Z5 = ONE.fit_transform(Z5).toarray()

Z6 = Z[:,6].reshape(-1,1)
Z6 = ONE.fit_transform(Z6).toarray()


Z = np.c_[Z6, Z5, Z4, Z1, Z2, Z3, Z[:, [23, 24]], Z[:, 26:32], Z[:, 33]]
print(Z)

Z_pred = classifier.predict(Z)

print(Z_pred)
