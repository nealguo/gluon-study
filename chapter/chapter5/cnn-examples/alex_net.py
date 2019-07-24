from mxnet import gluon, init, nd
from mxnet.gluon import nn

from utils import util

if __name__ == '__main__':
    net = nn.Sequential()
    # 使用较大的11×11窗口来捕获物体，同时使用步幅为4来较大幅度减少输出高和宽
    # 这里使用的输出通道数比LeNet中的大很多
    net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 连续3个卷积层，且使用更小的卷积窗口
            # 除了最后的卷积层外，进一步增大了输出通道数
            # 前两个卷积层后部使用池化层来减小输入的高和宽
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 这里全连接层的输出个数比LeNet中大数倍，使用丢弃层来缓解过拟合
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            # 输出层，对应Fashion-MNIST数据集的10种类别
            nn.Dense(10))
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    batch_size = 5
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size, resize=224)
    learning_rate = 0.01
    num_epochs = 5
    ctx = util.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    util.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
