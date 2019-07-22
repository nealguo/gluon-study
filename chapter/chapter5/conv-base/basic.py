from mxnet import autograd, nd
from mxnet.gluon import nn


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


if __name__ == '__main__':
    # 测试互相关函数corr2d
    X = nd.array([[0, 1, 2], [3., 4, 5], [6, 7, 8]])
    K = nd.array([[0, 1], [2, 3]])
    output = corr2d(X, K)
    print(output)

    # 测试卷积模型
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = nd.array([[1, -1]])  # 定义核数组
    Y = corr2d(X, K)
    print(Y)

    # 构造输出通道数为1，核数组形状为1×2的二维卷积层
    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()
    # 二维卷积层使用四维输入输出（样本/通道/高/宽）
    # 批量大小（批量重点样本数）和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        # 简单起见，这里忽略bias
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

    # 输出学习到的核数组
    K_learnt = conv2d.weight.data().reshape((1, 2))
    print(K_learnt)
