from utils import util
import math
from mxnet import nd


def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()


if __name__ == '__main__':
    # 学习率eta=0.4较小，随着s的累加而衰减，自变量在迭代后期的移动幅度较小
    eta = 0.4
    util.show_trace_2d(f_2d, util.train_2d(adagrad_2d))

    # 学习率eta=2较大，自变量更快速地逼近最优解
    eta = 2
    util.show_trace_2d(f_2d, util.train_2d(adagrad_2d))

    # 根据AdaGrad算法中的公式来实现
    features, labels = util.get_data_ch7()
    util.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)

    # 简洁实现
    util.train_gluon_ch7('adagrad', {'learning_rate': 0.1}, features, labels)
