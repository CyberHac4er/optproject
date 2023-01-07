import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt



class LogisticRegression():
    def __init__(self, l='m', lr=0.01, batchsize=32, epoch=5, method='SGD', max_iter=10000):
        self.lr = lr
        self.bs = batchsize
        self.epoch = epoch
        self.method = method
        self.max_iter = max_iter
        self.l = l
        self.accuracy=[]


    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        if self.l=='m':
            self.l = 1/X.shape[0]
        if self.method=='SGD':
            y_temp = self.predict(X)
            acc = self.acc(y, y_temp)
            for i in range(self.max_iter):
                if i%10==0:
                    print('iter:{}, Accuracy:{}'.format(i, acc))
                self.accuracy.append(acc)
                alpha = self.lr/np.sqrt(i+1)
                ind = np.random.randint(0, X.shape[0], self.bs)
                x_iter = X[ind]
                y_iter = y[ind]
                gd = 0
                for xi, yi in zip(x_iter, y_iter):
                    gd += -1/self.bs*((yi*xi)/(1+np.exp(yi*xi@self.theta))+self.l*self.theta)
                    # gd += -(yi * xi) / (1 + np.exp(yi * xi @ self.theta)) + self.l * self.theta
                # gd = gd.A[0]
                # gd = gd/(np.sqrt(gd@gd.reshape(-1, 1)))
                # print(gd@gd.reshape(-1, 1))
                self.theta-=(alpha*gd)
                yp = self.predict(X)
                new_acc = self.acc(y, yp)
                # print(np.linalg.norm(gd))
                if np.linalg.norm(gd)<1e-2:
                    break
                acc = new_acc
        elif self.method=='Newton':
            y_temp = self.predict(X)
            acc = self.acc(y, y_temp)
            for i in range(self.max_iter):
                alpha = self.lr
                # print('iter:{}, Accuracy:{}'.format(i, acc))
                self.accuracy.append(acc)
                if i % 10 == 0:
                    print('iter:{}, Accuracy:{}'.format(i, acc))
                gd_prime = 0
                for xi, yi in zip(X, y):
                    gd_prime += -1/X.shape[0]*(yi * xi) / (1 + np.exp(yi * xi @ self.theta)) + self.l * self.theta
                ind = np.random.randint(0, X.shape[0], self.bs)
                x_iter = X[ind]
                y_iter = y[ind]
                gd_prime2 = 0
                for xi, yi in zip(x_iter, y_iter):
                    gd_prime2+=-1/self.bs*((yi**2*xi@xi.reshape(-1, 1)*np.exp(yi*xi@self.theta))/(np.exp(yi*xi@self.theta)+1)**2+self.l)
                # print(gd_prime2)
                self.theta+=alpha*gd_prime/gd_prime2
                # print(self.theta)
                yp = self.predict(X)
                new_acc = self.acc(y, yp)
                # print(np.linalg.norm(gd))
                if np.linalg.norm(gd_prime) < 1e-1:
                    break
                acc = new_acc
        elif self.method=='RR':
            y_temp = self.predict(X)
            acc = self.acc(y, y_temp)
            for i in range(self.epoch):
                ind = np.arange(X.shape[0])
                np.random.shuffle(ind)
                alpha = self.lr/(i+1)
                for j in range(X.shape[0]//self.bs):
                    print('epoch:{}, iter:{}/{}, Accuracy:{}'.format(i, j+1, X.shape[0]//self.bs, acc))
                    self.accuracy.append(acc)
                    gd=0
                    for k in range(self.bs):
                        gd += -1 / self.bs * ((y[ind[self.bs*j+k]] * X[ind[self.bs*j+k]]) / (1 + np.exp(y[ind[self.bs*j+k]] * X[ind[self.bs*j+k]] @ self.theta)) + self.l * self.theta)
                    self.theta -= (alpha * gd)
                    yp = self.predict(X)
                    new_acc = self.acc(y, yp)
                    acc = new_acc

        return

    def predict(self, X):
        prob = 1/(1+np.exp(-X@self.theta))
        y_pred = np.where(prob>=0.5, 1, -1)
        return y_pred.reshape(-1, 1)

    def acc(self, y_test, y_pred):
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()
        return sum(y_test==y_pred)/len(y_test)

    def plot(self):
        x = np.arange(len(self.accuracy))
        plt.plot(x, self.accuracy)
        plt.ylabel('accuracy')
        plt.xlabel('t')
        plt.title(str(self.method))
        plt.savefig('plot/'+str(self.method)+'.png')
        plt.show()




