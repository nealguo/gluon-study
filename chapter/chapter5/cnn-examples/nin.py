from mxnet import gluon, init, nd
from mxnet.gluon import nn
from utils import util


def nin_block(num_channels, kernel_size, strides, padding):
    block = nn.Sequential()
    block.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
              nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
              nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return block


if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.5),
            # 10种类别，即标签类别数为10
            nin_block(10, kernel_size=3, strides=1, padding=1),
            # 全局平均池化层将窗口形状自动设置为输入的高和宽
            nn.GlobalAvgPool2D(),
            # 将四维的输出转为二维输出，形状为(batch_size,10)
            nn.Flatten())
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    lr = 0.1
    num_epochs = 5
    batch_size = 128
    ctx = util.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size, resize=224)
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
