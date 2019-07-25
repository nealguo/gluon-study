from mxnet import gluon, init, nd
from mxnet.gluon import nn

from utils import util


# 定义VGG块
def vgg_block(num_convs, num_channels):
    block = nn.Sequential()
    for _ in range(num_convs):
        block.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    block.add(nn.MaxPool2D(pool_size=2, strides=2))
    return block


# 定义VGG网络
def vgg(conv_arch):
    net = nn.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net


if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    net.initialize()
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    for block in net:
        X = block(X)
        print(block.name, 'output shape:\t', X.shape)

    ratio = 4
    lr = 0.05
    num_epochs = 5
    batch_size = 128
    ctx = util.try_gpu()
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size, resize=224)
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
