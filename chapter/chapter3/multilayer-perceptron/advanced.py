from mxnet import gluon, init, autograd
import sys


# 定义分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype("float32")
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 训练模型
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            # 累计训练中的损失量/准确次数
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        # 计算测试准确率
        test_acc = evaluate_accuracy(test_iter, net)

        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == '__main__':
    # 获取数据集
    mnist_train = gluon.data.vision.FashionMNIST(train=True)
    mnist_test = gluon.data.vision.FashionMNIST(train=False)

    # 批量读取数据
    batch_size = 256
    transformer = gluon.data.vision.transforms.ToTensor()
    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                       batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(mnist_test.transform_first(transformer),
                                      batch_size, shuffle=False,
                                      num_workers=num_workers)

    # 定义模型
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(256, activation="relu"), gluon.nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))

    # 定义损失函数
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # 定义优化算法
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.5})

    # 训练模型
    num_epochs = 5
    train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          None, None, trainer)
