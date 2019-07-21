from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.output = nn.Dense(10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))

    # 无需定义反向传播函数，系统将通过自动求梯度而自动生成反向传播所需的backward函数


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # 继承自Block的_children是OrderedDict类型，保证有序
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数rand_weight不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20))
        )
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数rand_weight，以及relu和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层，等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，使用asscalar函数返回标量并比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))


if __name__ == '__main__':
    # 测试自定义的MLP
    X = nd.random.uniform(shape=(2, 20))
    net = MLP()
    net.initialize()
    output = net(X)
    print(output)

    # 测试自定义的MySequential
    net2 = MySequential()
    net2.add(nn.Dense(256, activation='relu'))
    net2.add(nn.Dense(10))
    net2.initialize()
    output2 = net2(X)
    print(output2)

    # 测试自定义的FancyMLP
    net3 = FancyMLP()
    net3.initialize()
    output3 = net3(X)
    print(output3)

    # 测试自定义的NestMLP和FancyMLP
    net4 = nn.Sequential()
    net4.add(NestMLP(), nn.Dense(20), FancyMLP())
    net4.initialize()
    output4 = net4(X)
    print(output4)
