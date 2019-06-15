from mxnet import gluon, autograd, nd
import sys


# 定义丢弃算法
def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob


# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return nd.dot(H2, W3) + b3


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

    # 定义模型参数
    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddens1 = 256
    num_hiddens2 = 256

    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
    b1 = nd.zeros(num_hiddens1)
    W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
    b2 = nd.zeros(num_hiddens2)
    W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
    b3 = nd.zeros(num_outputs)

    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.attach_grad()

    # 训练和测试模型
    drop_prob1 = 0.2
    drop_prob2 = 0.5
    num_epochs = 5
    lr = 0.5
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params, lr)
