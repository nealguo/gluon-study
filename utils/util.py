from mxnet import gluon, autograd, init, image, nd
from mxnet.gluon import nn
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
        gluon.data.ArrayDataset(features, labels), batch_size, shuffle=True
    )
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
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


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format"""
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


class Benchmark():
    """Benchmark program"""

    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%s time: %.4f sec' % (self.prefix, time.time() - self.start))


def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent"""
    for p in params:
        p[:] = p - lr * p.grad / batch_size


def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes"""
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


class Residual(nn.Block):
    """The residual block"""

    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


def show_images(imgs, num_rows, num_cols, scale=2):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


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


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gluon.utils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)


def load_data_pikachu(batch_size, edge_size=256):
    """Download the pikachu dataest and then load into memory"""
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False)
    return train_iter, val_iter


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


def evaluate_accuracy(data_iter, net, ctx=[mxnet.cpu()]):
    """Evaluate accuracy of model on the given data set"""
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mxnet.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def _get_batch(batch, ctx):
    """Return features and labels on ctx"""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gluon.utils.split_and_load(features, ctx),
            gluon.utils.split_and_load(labels, ctx),
            features.shape[0])


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    """Train and evaluate a model"""
    print('training on ', ctx)
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum.asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc%.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))


def resnet18(num_classes):
    """The ResNet-18 model"""
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        block = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                block.add(Residual(num_channels))
        return block

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


def try_all_gpus():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU"""
    ctxes = []
    try:
        for i in range(16):
            ctx = mxnet.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mxnet.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mxnet.cpu()]
    return ctxes


def try_gpu():
    """If GPU is available, return gpu() else return cpu()"""
    try:
        ctx = mxnet.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mxnet.base.MXNetError:
        ctx = mxnet.cpu()
    return ctx
