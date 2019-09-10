from utils import util
import math
from mxnet import nd


def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_rmsprop_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()


if __name__ == '__main__':
    # RMSProp算法将梯度按元素平方做指数加权移动平均
    # 使得每个个元素的学习率在迭代过程中就不再一直降低或不便
    eta, gamma = 0.4, 0.9
    util.show_trace_2d(f_2d, util.train_2d(rmsprop_2d))

    # 根据RMSProp算法中的公式来实现
    features, labels = util.get_data_ch7()
    util.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9}, features, labels)

    # 简洁实现，超参gamma通过gamma2指定
    util.train_gluon_ch7('rmsprop', {'learning_rate': 0.01, 'gamma2': 0.9}, features, labels)
