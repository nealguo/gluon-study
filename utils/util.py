from mxnet import gluon, autograd, init, nd
import numpy as np
import mxnet
import sys
import time
import os

# need to install IPython and module named 'ipykernel'
# pip install ipython
# pip install ipykernel
from IPython import display
from matplotlib import pyplot as plt


def linreg(X, w, b):
    """Linear regression"""
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def show_trace_2d(f, res):
    """Show the trace of 2d variables during optimization"""
    x1, x2 = zip(*res)
    set_figsize()
    plt.plot(x1, x2, '-o', color='#ff7f0e')
    x1 = np.arange(-5.5, 1.0, 0.1)
    x2 = np.arange(min(-3.0, min(x2) - 1), max(1.0, max(x2) + 1), 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def train_2d(trainer):
    """Optimize the objective function of 2d variables with a customized trainer"""
    x1, x2 = -5, -2
    s_x1, s_x2 = 0, 0
    res = [(x1, x2)]
    for i in range(20):
        x1, x2, s_x1, s_x2 = trainer(x1, x2, s_x1, s_x2)
        res.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (20, x1, x2))
    return res


def get_data_ch7():
    """Get the data set used in Chapter 7"""
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:, :-1]), nd.array(data[:, -1])


def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    """Train a linear regression model"""
    net, loss = linreg, squared_loss
    w, b = nd.random.normal(scale=0.01, shape=(features.shape[1], 1)), nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(features, labels), batch_size, shuffle=True
    )
    start = time.time()
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    end = time.time()
    print('loss: %f, %f sec per epoch' % (ls[-1], end - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    """Train a linear regression model with a given Gluon trainer."""
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gluon.loss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(features, labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),
                            trainer_name, trainer_hyperparams)
    start = time.time()
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    end = time.time()
    print('loss: %f, %f sec per epoch' % (ls[-1], end - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot"""
    display.set_matplotlib_formats('svg')


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
