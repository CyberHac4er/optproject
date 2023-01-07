import numpy as np
from gradient_method import LogisticRegression
import matplotlib.pyplot as plt
import time



X_o = np.load('data/data_set1.npy')
y = X_o[:8000, 2]
y_test = X_o[8000:, 2]
X = X_o[:8000,:2]
X_test = X_o[8000:, :2]
# mu = np.mean(X, axis=0)
# sigma = np.std(X, axis=0)
# X = (X-mu)/sigma
X = np.concatenate([np.full([X.shape[0], 1], 1), X], axis=1)
X_test = np.concatenate([np.full([X_test.shape[0], 1], 1), X_test], axis=1)
y = np.where(y==0, -1, 1)
y_test = np.where(y_test==0, -1, 1)

lr = LogisticRegression(lr=0.01, method='SGD', max_iter=1000)
lr.fit(X, y)
y_pred = lr.predict(X_test)
print('test accuracy is', lr.acc(y_test, y_pred))
lr.plot()







