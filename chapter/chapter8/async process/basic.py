from mxnet import nd
import time


class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%s time: %.4f sec' % (self.prefix, time.time() - self.start))


if __name__ == '__main__':
    x = nd.random.uniform(shape=(2000, 2000))

    # 异步缓存
    with Benchmark('Workloads are queued'):
        y = nd.dot(x, x).sum()

    # 同步计算
    # 验证同步函数 print/wait_to_read/waitall/asnumpy/asscalar

    with Benchmark('Workloads are finished'):
        # print()会让前端线程等待MXNet后端取出y的计算表达式并计算返回结果
        # 即print()有同步等待的作用
        print('sum=', y)

    with Benchmark('wait_to_read'):
        # y是一个NDArray，即计算结果会保存在y这个NDArray中
        y = nd.dot(x, x)
        # 使用wait_to_read()让前端等待y的计算结果返回
        y.wait_to_read()

    with Benchmark('waitall'):
        y = nd.dot(x, x)
        z = nd.dot(x, x)
        # 使用waitall()让前端等待前面所有计算机过完成
        nd.waitall()

    with Benchmark('asnumpy'):
        y = nd.dot(x, x)
        # asnumpy()会将NDArray转换为其他不支持异步计算的数据结构
        # 这样的转换会让前端等待计算结果
        y.asnumpy()

    with Benchmark('asscalar'):
        y = nd.dot(x, x)
        # asscalar()会将NDArray转换为其他不支持异步计算的数据结构
        # 这样的转换会让前端等待计算结果
        y.norm().asscalar()

    # 异步计算
    with Benchmark('synchronous'):
        for _ in range(1000):
            y = x + 1
            # for循环内使用同步函数wait_to_read时，每次赋值不使用异步计算
            y.wait_to_read()

    with Benchmark('asynchronous'):
        for _ in range(1000):
            y = x + 1
        # for循环外使用同步函数waitall时，则使用异步计算
        nd.waitall()
