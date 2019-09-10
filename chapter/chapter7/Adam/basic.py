from utils import util
from mxnet import nd


def init_adam_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return ((v_w, s_w), (v_b, s_b))


def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (s_bias_corr.sqrt() + eps)
    hyperparams['t'] += 1


if __name__ == '__main__':
    # 根据Adam算法的公式来实现
    features, labels = util.get_data_ch7()
    util.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)

    # 简洁实现
    util.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)
