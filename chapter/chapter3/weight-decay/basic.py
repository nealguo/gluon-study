from mxnet import gluon, autograd, nd


# 初始化模型参数
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


# 定义L2范数惩罚项
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 定义模型
def lin_reg(X, w, b):
    return nd.dot(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 定义模型训练和测试
def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    print("L2 norm of w:", w.norm().asscalar())


if __name__ == '__main__':
    # 准备训练和测试数据
    n_train = 20
    n_test = 100
    num_inputs = 200
    true_w = nd.ones((num_inputs, 1)) * 0.01
    true_b = 0.05

    features = nd.random.normal(shape=(n_train + n_test, num_inputs))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    train_features = features[:n_train, :]
    test_features = features[n_train:, :]
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    # 开始训练和测试
    batch_size = 1
    num_epochs = 100
    lr = 0.003
    net = lin_reg
    loss = squared_loss
    data_set = gluon.data.ArrayDataset(train_features, train_labels)
    train_iter = gluon.data.DataLoader(data_set, batch_size, shuffle=True)

    # 过拟合
    fit_and_plot(lambd=0)
    # 使用权重衰减
    fit_and_plot(lambd=3)
