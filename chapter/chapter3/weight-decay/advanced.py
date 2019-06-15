from mxnet import gluon, init, autograd, nd


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义模型训练和测试
def fit_and_plot_gluon(lambd):
    # 定义模型
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize(init.Normal(sigma=1))

    # 对权重参数衰减
    trainer_w = gluon.Trainer(net.collect_params(".*weight"), "sgd", {"learning_rate": lr, "wd": lambd})
    # 不对偏差参数衰减
    trainer_b = gluon.Trainer(net.collect_params(".*bias"), "sgd", {"learning_rate": lr})

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print("L2 norm of w:", net[0].weight.data().norm().asscalar())


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
    loss = squared_loss
    data_set = gluon.data.ArrayDataset(train_features, train_labels)
    train_iter = gluon.data.DataLoader(data_set, batch_size, shuffle=True)

    # 过拟合
    fit_and_plot_gluon(lambd=0)
    # 使用权重衰减
    fit_and_plot_gluon(lambd=3)
