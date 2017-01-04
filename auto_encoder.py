import chainer
from chainer import Chain
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import numpy as np
from sklearn import datasets

#iris = datasets.load_iris()
#print(iris.data.shape) #(500, 4)
#xtrain = iris.data.astype(np.float32)

train, test = chainer.datasets.get_mnist(withlabel=True, ndim=1)
# train is data with label tuple, xtrain contains only data

xtrain = np.asarray(list(map(lambda x: x[0], train)))
labels = np.array(train)[:,1]

num_index = [0] * 9
for i in range (0,9):
  num_index[i] = np.where(labels == i)

class AutoEncoder(Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__(
            l1 = L.Linear(784,20),
            l2 = L.Linear(20,784)
        )
    def __call__(self, x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)
    def fwd(self, x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv

model = AutoEncoder()
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

n = 6000 # sample number
batch_size = 300
for j in  range(3000):
    shuffle_index = np.random.permutation(n)
    for i in range(0, n, batch_size):
        x = Variable(xtrain[shuffle_index[i:(i+batch_size) if (i+batch_size) < n else n]])
        model.zerograds()
        loss = model(x)
        loss.backward()
        optimizer.update()

import matplotlib.pyplot as plt
x = Variable(xtrain)
yt = F.sigmoid(model.l1(x))
ans = yt.data

print("training ended")

#for i in range(0, 9):
#    ansx = ans[0:50, 0]
#    ansy = ans[0:50, 1]
#    plt.scatter(ansx, ansy)
#
#plt.savefig('graph.png', bbox_inches='tight')
