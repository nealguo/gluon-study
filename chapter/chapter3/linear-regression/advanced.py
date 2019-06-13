from mxnet import gluon, init, autograd, nd

if __name__ == '__main__':
    # 定义模型
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))

    # 初始化模型参数
    net.initialize(init.Normal(sigma=0.01))

    # 定义损失函数
    loss = gluon.loss.L2Loss()

    # 定义优化算法
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})

    # 生成数据集
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(1000, 2))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    # 读取数据
    batch_size = 10
    data_set = gluon.data.ArrayDataset(features, labels)
    data_iter = gluon.data.DataLoader(data_set, batch_size, shuffle=True)

    # 训练模型
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print("epoch %d, loss: %f" % (epoch, l.mean().asnumpy()))

    # 打印结果
    print(true_w)
    print(net[0].weight.data())

    print(true_b)
    print(net[0].bias.data())
