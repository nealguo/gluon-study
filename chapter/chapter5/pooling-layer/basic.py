from mxnet import nd
from mxnet.gluon import nn


# 定义二维池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


if __name__ == '__main__':
    # 测试自定义池化层
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # 最大池化层
    print(pool2d(X, (2, 2)))
    # 平均池化层
    print(pool2d(X, (2, 2), 'avg'))

    # 使用MaxPool2D并设置步幅和池化窗口形状都为3×3
    X = nd.arange(16).reshape((1, 1, 4, 4))
    pool2d = nn.MaxPool2D(3)
    # 池化层没有模型参数，无需调用参数初始化函数
    print(pool2d(X))
    # 设置池化窗口为3×3、填充为1、步幅为2
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))
    # 设置池化窗口为2×3、高和宽两侧的填充分别为1和2、高和宽两侧的步幅分别为2和3
    pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
    print(pool2d(X))

    # 测试对多通道输入的池化操作
    # 将数值X和X+1在通道维上连结起来构造通道数为2的输入
    X = nd.concat(X, X + 1, dim=1)
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))
