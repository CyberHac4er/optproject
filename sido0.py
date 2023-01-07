import numpy as np
import scipy.io
from gradient_method import LogisticRegression
from scipy import sparse



mat_contents = scipy.io.loadmat('data/sido0/sido0_train.mat')
A = mat_contents['A']

mat_contents = scipy.io.loadmat('data/sido0/sido0_train_label.mat')
b = mat_contents['b']

lr = LogisticRegression(lr=0.01)
lr.fit(A, b)