from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
import time
import os
import sys
import subprocess
# need to install the psutil
# pip install psutil
import psutil


def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print('batch %d, time %f sec' % (i + 1, time.time() - start))


# 获取内存的使用情况（仅仅Linux/MacOS上可用）
def get_mem():
    if sys.platform.startswith("win"):
        res = psutil.Process(os.getpid()).memory_info().rss
        return int(str(res)) / 1e6
    else:
        res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
        return int(str(res).split()[15]) / 1e6


if __name__ == '__main__':
    # 定义多层感知机模型、优化算法和损失函数
    net = nn.Sequential()
    net.add(nn.Dense(2048, activation='relu'),
            nn.Dense(512, activation='relu'),
            nn.Dense(1))
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
    loss = gluon.loss.L2Loss()

    # 先试运行一次，让系统把net的参数初始化
    for X, y in data_iter():
        loss(y, net(X)).wait_to_read()
        break

    # 每个小批量都同步计算，减少内存使用，增加了训练时间
    l_sum, mem = 0, get_mem()
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l_sum += l.mean().asscalar()  # 使用同步函数asscalar()
        l.backward()
        trainer.step(X.shape[0])
    nd.waitall()
    print('increased memory: %f MB' % (get_mem() - mem))

    # 每个小批量都异步计算，增加内存使用，减少了训练时间
    mem = get_mem()
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
    nd.waitall()
    print('increased memory: %f MB' % (get_mem() - mem))
