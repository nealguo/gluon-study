from mxnet import gluon, init, nd
from mxnet.gluon import nn

from utils import util

if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    batch_size = 256
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size)
    learning_rate = 0.9
    num_epochs = 5
    ctx = util.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
