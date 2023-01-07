from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.io


mat_contents = scipy.io.loadmat('data/a9a/a9a_train.mat')
A = mat_contents['A']
A = A[:, :-1]


mat_contents = scipy.io.loadmat('data/a9a/a9a_train_label.mat')
b = mat_contents['b'].flatten()

mat_contents = scipy.io.loadmat('data/a9a/a9a_test.mat')
A_test = mat_contents['A']

mat_contents = scipy.io.loadmat('data/a9a/a9a_test_label.mat')
b_test = mat_contents['b']

lr = LogisticRegression(solver='liblinear')
lr.fit(A, b)
print(lr.score(A_test, b_test))

