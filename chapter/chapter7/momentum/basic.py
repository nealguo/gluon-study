from utils import util
from mxnet import nd


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


# 使用动量法
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
        p[:] -= v


if __name__ == '__main__':
    # 学习率eta=0.4，x2方向上比x1方向上移动幅度更大
    eta = 0.4
    util.show_trace_2d(f_2d, util.train_2d(gd_2d))

    # 学习率eta=0.6，学习率过大使得在x2方向上不断越过最优解兵逐渐发散
    eta = 0.6
    util.show_trace_2d(f_2d, util.train_2d(gd_2d))

    # 采用动量法，使得在x2方向上移动更加平滑且在x1方向上更快逼近最优解
    eta, gamma = 0.4, 0.5
    util.show_trace_2d(f_2d, util.train_2d(momentum_2d))

    # 采用动量法，较大的学习率eta=0.6也不会使得自变量发散
    eta, gamma = 0.6, 0.5
    util.show_trace_2d(f_2d, util.train_2d(momentum_2d))

    # 学习率lr=0.2，momentum=0.5，目标函数在后期迭代变化较平衡
    features, labels = util.get_data_ch7()
    util.train_ch7(sgd_momentum, init_momentum_states(),
                   {'lr': 0.02, 'momentum': 0.5}, features, labels)

    # 增大超参momentum=0.9，目标函数在后期迭代中变化不够平滑
    util.train_ch7(sgd_momentum, init_momentum_states(),
                   {'lr': 0.02, 'momentum': 0.9}, features, labels)

    # 相对momentum=0.2(即2倍小批量梯度)的情况，momentum=0.9(即10被小批量梯度)减小学习率为原来的1/5
    # 目标函数在后期迭代变化较平衡
    util.train_ch7(sgd_momentum, init_momentum_states(),
                   {'lr': 0.004, 'momentum': 0.9}, features, labels)

    # 简洁实现方式
    util.train_gluon_ch7('sgd', {'learning_rate': 0.004, 'momentum': 0.9}, features, labels)
