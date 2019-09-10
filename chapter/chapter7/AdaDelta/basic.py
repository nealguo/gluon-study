from utils import util
from mxnet import nd


def init_adadelta_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    delta_w = nd.zeros((features.shape[1], 1))
    delta_b = nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))


def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g


if __name__ == '__main__':
    # 根据AdaDelta算法中的公式来实现
    features, labels = util.get_data_ch7()
    util.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)

    # 简洁实现
    util.train_gluon_ch7('adadelta', {'rho': 0.9}, features, labels)
