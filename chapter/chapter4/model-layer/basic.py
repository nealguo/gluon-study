from mxnet import nd
from mxnet.gluon import nn


class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


class MyDense(nn.Block):
    # out_units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, out_units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, out_units))
        self.bias = self.params.get('bias', shape=(out_units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


if __name__ == '__main__':
    # 不含模型参数的自定义层
    layer = CenteredLayer()
    print(layer(nd.array([1, 2, 3, 4, 5])))

    net = nn.Sequential()
    net.add(nn.Dense(128),
            CenteredLayer())
    net.initialize()
    y = net(nd.random.uniform(shape=(4, 8)))
    print(y.mean().asscalar())

    # 含有模型参数的自定义层
    dense = MyDense(out_units=3, in_units=5)
    print(dense.params)
    dense.initialize()
    print(dense(nd.random.uniform(shape=(2, 5))))

    net2 = nn.Sequential()
    net2.add(MyDense(out_units=8, in_units=64),
             MyDense(out_units=1, in_units=8))
    net2.initialize()
    print(net2(nd.random.uniform(shape=(2, 64))))
