from mxnet import gluon, autograd, nd
import pandas as pd


# 定义训练模型（线性回归）
def get_net():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net


# 定义损失函数（对数均方根误差）
def log_rmse(net, features, labels):
    # 将小于1的值设为1，使得取对数时数值更稳定
    clipped_preds = nd.clip(net(features), 1, float("inf"))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


# 模型训练
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": learning_rate, "wd": weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 定义K折交叉验证，返回第i折交叉验证所需训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid


# 定义K折交叉验证，返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data,
                                   num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print("fold %d, train rmse %f, valid rmse %f" % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


# 训练并预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, learning_rate, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, learning_rate, weight_decay, batch_size)
    print("train rmse %f" % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    # 获取数据集
    train_data = pd.read_csv("../data/kaggle_house_pred_train.csv")
    test_data = pd.read_csv("../data/kaggle_house_pred_test.csv")
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 预处理数据
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    n_train = train_data.shape[0]
    train_features = nd.array(all_features[:n_train].values)
    test_features = nd.array(all_features[n_train:].values)
    train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

    # 损失函数选择
    loss = gluon.loss.L2Loss()

    # 模型选择
    k = 5
    num_epochs = 100
    learning_rate = 5
    weight_decay = 0
    batch_size = 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
                              learning_rate, weight_decay, batch_size)
    print("%d-fold validation: avg train rmse %f, avg valid rmse %f" % (k, train_l, valid_l))

    # 重新训练并保存预测结果
    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, learning_rate, weight_decay, batch_size)
