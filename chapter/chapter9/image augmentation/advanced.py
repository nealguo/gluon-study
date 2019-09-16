from mxnet import gluon, init, nd
from utils import util
import mxnet as mx
import sys


def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):  # 假设一台机器上GPU的数量不超过16
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, ctx, net = 256, try_all_gpus(), util.resnet18(10)
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    util.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)


if __name__ == '__main__':
    util.show_images(gluon.data.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8)
    util.plt.show()

    flip_aug = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.RandomFlipLeftRight(),
        gluon.data.vision.transforms.ToTensor()])

    no_aug = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor()])

    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_with_data_aug(flip_aug, no_aug)
