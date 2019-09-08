import numpy as np
import time
from mxnet import autograd, gluon, nd
from utils import util


def get_data_ch7():
    data = np.genfromtxt("../data/airfoil_self_noise.dat", delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])


def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad


def train_ch7(trainer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = util.linreg, util.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
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
                l = loss(net(X, w, b), y).mean()  # 使用平均损失
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    end = time.time()

    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], end - start))
    util.set_figsize()
    util.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    util.plt.xlabel('epoch')
    util.plt.ylabel('loss')
    util.plt.show()


def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


if __name__ == '__main__':
    features, labels = get_dataa_ch7()
    print(features.shape)

    # 批量大小为1500（样本总数），等价于梯度下降
    train_sgd(1, 1500, 6)
    # 批量大小为1，等价于随机梯度下降
    train_sgd(0.005, 1)
    # 批量大小为10，等价于小批量随机梯度下降
    train_sgd(0.05, 10)
