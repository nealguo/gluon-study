from mxnet import init, nd
from mxnet.gluon import nn


# 自定义初始化方法
class MyInit(init.Initializer):
    # 只需实现_init_weight函数
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 令权重有一半概率初始化为0，有一半概率初始化为[-10,-5]和[5,10]两个区间里均匀分布的随机数
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    # net首次被初始化，使用默认初始化方式
    net.initialize()

    X = nd.random.uniform(shape=(2, 20))
    Y = net(X)  # 前向计算

    # net再次被初始化，使用init模块中正太分布初始化方法
    net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
    print(net[0].weight.data()[0])

    # net再次被初始化，使用init模块的常数来初始化权重参数
    net.initialize(init=init.Constant(1), force_reinit=True)
    print(net[0].weight.data()[0])

    # net再次被初始化，使用init模块的Xavier随机初始化方法
    net.initialize(init=init.Xavier(), force_reinit=True)
    print(net[0].weight.data()[0])

    # net再次被初始化，使用自定义的初始化方法
    net.initialize(init=MyInit(), force_reinit=True)
    print(net[0].weight.data()[0])

    # 测试共享模型参数，第二隐藏层和第三隐藏层共享模型参数
    net2 = nn.Sequential()
    second = nn.Dense(8, activation='relu')
    third = nn.Dense(8, activation='relu', params=second.params)
    net2.add(nn.Dense(8, activation='relu'),
             second,  # 第二隐藏层
             third,  # 第三隐藏层
             nn.Dense(10))
    net2.initialize()
    X2 = nd.random.uniform(shape=(2, 20))
    Y2 = net2(X2)
    print(net2[1].weight.data()[0] == net2[2].weight.data()[0])
