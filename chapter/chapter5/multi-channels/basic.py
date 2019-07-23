from mxnet import nd


# 二维互相关
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 对多通道输入的二维互相关
def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历
    # 然后使用*将结果列表变成add_n函数的位置参数来进行相加
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])


# 对多通道输入和多通道输出的二维互相关
def corr2d_multi_in_out(X, K):
    # 对K的第0维（通道维）遍历，每次与输入X做互相关计算
    # 所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


# 对多通道输入和多通道输出使用1×1卷积核的二维互相关
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层的矩阵乘法
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == '__main__':
    X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    print(corr2d_multi_in(X, K))

    # 核数组K与K+1和K+2连接起来构造一个输出通道数为3的卷积核
    # K+1即K中每个元素加1，K+2同理
    K = nd.stack(K, K + 1, K + 2)
    print(K.shape)
    print(corr2d_multi_in_out(X, K))

    # 做1×1卷积时，corr2d_multi_in_out_1x1和corr2d_multi_in_out等价
    X = nd.random.uniform(shape=(3, 3, 3))
    K = nd.random.uniform(shape=(2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    print((Y1 - Y2).norm().asscalar() < 1e-6)
