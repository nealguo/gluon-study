import mxnet as mx
from mxnet import nd
from utils import util


def run(x):
    return [nd.dot(x, x) for _ in range(10)]


def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]


if __name__ == '__main__':
    x_cpu = nd.random.uniform(shape=(2000, 2000))
    x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))

    run(x_cpu)  # 预热开始
    run(x_gpu)
    nd.waitall()  # 预热结束

    # CPU和GPU的并行计算
    with util.Benchmark('Run on CPU'):
        run(x_cpu)
        nd.waitall()

    with util.Benchmark('Run on GPU'):
        run(x_gpu)
        nd.waitall()

    with util.Benchmark('Run on both CPU and GPU in parallel'):
        run(x_cpu)
        run(x_gpu)
        nd.waitall()

    # 计算和通信的并行计算
    with util.Benchmark('Run on GPU'):
        y = run(x_gpu)
        nd.waitall()

    with util.Benchmark('Then copy to CPU'):
        copy_to_cpu(y)
        nd.waitall()

    with util.Benchmark('Run and copy in parallel'):
        y = run(x_gpu)
        copy_to_cpu(y)
        nd.waitall()
