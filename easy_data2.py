import numpy as np
from gradient_method import LogisticRegression
import time

X_o = np.load('data/data_set2_large.npy')
X = X_o[:1000,:2]
X_test = X_o[1000:1200, :2]
y = X_o[:1000, 2]
y_test = X_o[100000:120000, 2]
X = np.concatenate([np.full([X.shape[0], 1], 1), X], axis=1)
X_test = np.concatenate([np.full([X_test.shape[0], 1], 1), X_test], axis=1)
y = np.where(y==0, -1, 1)
y_test = np.where(y_test==0, -1, 1)

start = time.time()
lr = LogisticRegression(lr=0.01, method='SGD', max_iter=1000)
lr.fit(X, y)
y_pred = lr.predict(X_test)
print('test accuracy is', lr.acc(y_test, y_pred))
end = time.time()
lr.plot()



