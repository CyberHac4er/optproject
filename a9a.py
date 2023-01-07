import numpy as np
import scipy.io
from gradient_method import LogisticRegression
from scipy import sparse

#load data
mat_contents = scipy.io.loadmat('data/a9a/a9a_train.mat')
A = mat_contents['A']
A = A[:,:-1]
iden = np.full(A.shape[0], 1)
A = sparse.hstack((A, iden[:,None]))
A = A.tocsr()

mat_contents = scipy.io.loadmat('data/a9a/a9a_train_label.mat')
b = mat_contents['b']

mat_contents = scipy.io.loadmat('data/a9a/a9a_test.mat')
A_test = mat_contents['A']
iden = np.full(A_test.shape[0], 1)
A_test = sparse.hstack((A_test, iden[:,None]))
A_test = A_test.tocsr()

mat_contents = scipy.io.loadmat('data/a9a/a9a_test_label.mat')
b_test = mat_contents['b']

lr = LogisticRegression(lr=0.01, l=0.001)
lr.fit(A, b)
b_pred = lr.predict(A_test)
print('test accuracy is', lr.acc(b_test, b_pred))