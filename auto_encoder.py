import chainer
import argparse
from chainer import Chain
from chainer import Variable
from chainer import training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import numpy as np
from sklearn import datasets

#iris = datasets.load_iris()
#print(iris.data.shape) #(500, 4)
#xtrain = iris.data.astype(np.float32)

train, test = chainer.datasets.get_mnist(withlabel=True, ndim=1)
# train is data with label tuple, xtrain contains only data

#xtrain = np.asarray(list(map(lambda x: x[0], train)))
#labels = np.array(train)[:,1]
#
#num_index = [0] * 9
#for i in range (0,9):
#  num_index[i] = np.where(labels == i)
#
class AutoEncoder(Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__(
            l1 = L.Linear(784,20),
            l2 = L.Linear(20,784)
        )
    def __call__(self, x, param):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)
    def fwd(self, x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv

parser = argparse.ArgumentParser(description='AutoEncoder: ')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

model = AutoEncoder()
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle = False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())

trainer.run()

exit()
#n = 6000 # sample number
#batch_size = 300
#for j in  range(3000):
#    shuffle_index = np.random.permutation(n)
#    for i in range(0, n, batch_size):
#        x = Variable(xtrain[shuffle_index[i:(i+batch_size) if (i+batch_size) < n else n]])
#        model.zerograds()
#        loss = model(x)
#        loss.backward()
#        optimizer.update()
#
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
