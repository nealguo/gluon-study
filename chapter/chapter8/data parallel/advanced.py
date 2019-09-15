from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
from utils import util
import mxnet as mx
import time


def result18(num_classes):
    def resnet_block(num_channels, num_residuals, first_block=False):
        block = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.add(util.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                block.add(util.Residual(num_channels))
        return block

    net = nn.Sequential()
    # 这里使用较小的卷积核、步幅和填充，并去掉最大池化层
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = util.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gluon.utils.split_and_load(X, ctx)
            gpu_ys = gluon.utils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y) for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = util.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test acc %.2f' % (epoch + 1, train_time, test_acc))


if __name__ == '__main__':
    net = result18(10)
    ctx = [mx.gpu(0), mx.gpu(1)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)

    x = nd.random.uniform(shape=(4, 1, 28, 28))
    gpu_x = gluon.utils.split_and_load(x, ctx)
    print(net(gpu_x[0]))
    print(net(gpu_x[1]))

    # 显示不同显卡的显存上的模型参数
    weight = net[0].params.get('weight')
    try:
        weight.data()
    except RuntimeError:
        print('not initialized on ', mx.cpu())
    print(weight.data(ctx[0])[0])
    print(weight.data(ctx[1])[0])

    # 单个GPU上的训练
    train(num_gpus=1, batch_size=256, lr=0.1)
    # 两块GPU上的训练
    train(num_gpus=2, batch_size=512, lr=0.2)
