from mxnet import gluon, autograd, nd
import mxnet
import sys
import time
import os


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
    '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Load the fashion mnist dataset into memory"""
    root = os.path.expanduser(root)  # 展开用户路径'~'
    transformer = []
    if resize:
        transformer += [gluon.data.vision.transforms.Resize(resize)]
    transformer += [gluon.data.vision.transforms.ToTensor()]
    transformer = gluon.data.vision.transforms.Compose(transformer)

    # 获取数据集
    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False)

    # 批量读取数据
    num_workers = 0 if sys.platform.startswith("win32") else 4
    train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                       batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(mnist_test.transform_first(transformer),
                                      batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    """Train and evaluate a model with CPU or GPU"""
    print('training on', ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)

            # 累计训练中的损失量/准确次数
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        # 计算测试准确率
        test_acc = evaluate_accuracy(test_iter, net, ctx)

        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net, ctx):
    """Evaluate accuracy of model on the given data set"""
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果ctx为GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def try_gpu():
    """If GPU is available, return gpu() else return cpu()"""
    try:
        ctx = mxnet.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mxnet.base.MXNetError:
        ctx = mxnet.cpu()
    return ctx
