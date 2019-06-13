from mxnet import autograd, nd
import random


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


# STEP-1 定义模型
def lin_reg(X, w, b):
    return nd.dot(X, w) + b


# STEP-2 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# STEP-3 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# STEP-4 训练模型
if __name__ == "__main__":
    # 初始化模型参数
    W = nd.random.normal(scale=0.01, shape=(2, 1))
    b = nd.zeros(shape=(1,))

    # 创建模型参数的梯度
    W.attach_grad()
    b.attach_grad()

    # 初始化学习率/迭代次数
    lr = 0.03
    num_epochs = 3
    net = lin_reg
    loss = squared_loss

    # 生成数据集
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(1000, 2))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    # 训练模型
    batch_size = 10
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, W, b), y)
            l.backward()
            sgd([W, b], lr, batch_size)
        train_l = loss(net(features, W, b), labels)
        print("epoch %d, loss %f" % (epoch + 1, train_l.mean().asnumpy()))

    # 打印结果
    print(true_w)
    print(W)

    print(true_b)
    print(b)
