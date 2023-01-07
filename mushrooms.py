import numpy as np
import scipy.io
from gradient_method import LogisticRegression
from scipy import sparse
import time


mat_contents = scipy.io.loadmat('data/mushrooms/mushrooms_train.mat')
A = mat_contents['A']
iden = np.full(A.shape[0], 1)
A = sparse.hstack((A, iden[:,None]))
A = A.tocsr()

mat_contents = scipy.io.loadmat('data/mushrooms/mushrooms_train_label.mat')
b = mat_contents['b']
start = time.time()
lr = LogisticRegression(lr=0.01, method='Newton', max_iter=100)
lr.fit(A, b)
end = time.time()
print(lr.method, ':', end-start)
