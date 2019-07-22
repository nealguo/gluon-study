from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


if __name__ == '__main__':
    # 使用自定义模型MLP并初始化
    net = MLP()
    net.initialize()
    X = nd.random.uniform(shape=(2, 20))
    output = net(X)
    print(output)

    # 保存net模型的参数到文件
    filename = 'mlp.params'
    net.save_params(filename)

    # 从文件加载模型参数到net2
    net2 = MLP()
    net2.load_params(filename)
    output2 = net2(X)
    print(output2)

    # 比较两个模型的输出是否一致
    print(output == output2)
