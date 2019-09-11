from mxnet import nd, sym
from mxnet.gluon import nn
import time


def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net


def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall()  # 等待所有计算完成方便记时
    end = time.time()
    return end - start


class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('F:', F)
        print('x:', x)
        x = F.relu(self.hidden(x))
        print('hidden:', x)
        return self.output(x)


if __name__ == '__main__':
    x = nd.random.normal(shape=(1, 512))
    net = get_net()
    print(net(x))

    net.hybridize()
    print(net(x))

    net = get_net()
    print('before hybridizing:%.4f sec' % (benchmark(net, x)))
    net.hybridize()
    print('after hybridizing:%.4f sec' % (benchmark(net, x)))

    # 将符号式程序和模型参数保存到磁盘
    net.export('my_mlp')

    # 对于调用了hybridize()后的模型
    # 输入Symbol类型的变量，net(x)会返回Symbol类型的结果
    x = sym.var('data')
    print(net(x))

    # F使用NDArray
    net = HybridNet()
    net.initialize()
    x = nd.random.normal(shape=(1, 4))
    print(net(x))

    # F变成了Symbol
    net.hybridize()
    print(net(x))
