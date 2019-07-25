from mxnet import gluon, init
from mxnet.gluon import nn

from utils import util

if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.Dense(84),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.Dense(10))

    lr, num_epochs, batch_size, ctx = 1.0, 5, 256, util.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size)
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
