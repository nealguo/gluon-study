import numpy as np
import time
from mxnet import autograd, gluon, init, nd
from utils import util


def get_data_ch7():
    data = np.genfromtxt("../data/airfoil_self_noise.dat", delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])


def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型
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

    # 创建Trainer实例来迭代模型参数
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
    start = time.time()
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)  # 在Trainer实例中做梯度平均
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    end = time.time()

    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], end - start))
    util.set_figsize()
    util.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    util.plt.xlabel('epoch')
    util.plt.ylabel('loss')
    util.plt.show()


if __name__ == '__main__':
    features, labels = get_data_ch7()
    print(features.shape)

    train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
