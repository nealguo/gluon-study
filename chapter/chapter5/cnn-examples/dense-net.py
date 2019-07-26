from mxnet import gluon, init, nd
from mxnet.gluon import nn

from utils import util


# 定义卷积块
def conv_block(num_channels):
    block = nn.Sequential()
    block.add(nn.BatchNorm(), nn.Activation('relu'),
              nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return block


# 定义过渡块
def transition_block(num_channels):
    block = nn.Sequential()
    block.add(nn.BatchNorm(), nn.Activation('relu'),
              nn.Conv2D(num_channels, kernel_size=1),
              nn.AvgPool2D(pool_size=2, strides=2))
    return block


# 定义稠密层
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            # 在通道维上将输入和输出连结
            X = nd.concat(X, Y, dim=1)
        return X


if __name__ == '__main__':
    block = DenseBlock(2, 10)
    block.initialize()
    X = nd.random.uniform(shape=(4, 3, 8, 8))
    Y = block(X)
    print(Y.shape)

    block = transition_block(10)
    block.initialize()
    print(block(Y).shape)

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, padding=3, strides=2),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, padding=1, strides=2))

    num_channels, growth_rate = 64, 32  # num_channels为当前通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(transition_block(num_channels))
    net.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.GlobalAvgPool2D(), nn.Dense(10))

    lr, num_epochs, batch_size, ctx = 0.1, 5, 256, util.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size, resize=96)
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
