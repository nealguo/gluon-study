from utils import util
from mpl_toolkits import mplot3d
import numpy as np


def f(x):
    return x * np.cos(np.pi * x)


def f2(x):
    return x ** 3


if __name__ == '__main__':
    # 局部最小值，全局最小值
    util.set_figsize((4.5, 2.5))
    x = np.arange(-1.0, 2.0, 0.1)
    fig, = util.plt.plot(x, f(x))
    fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                      arrowprops=dict(arrowstyle='->'))
    fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                      arrowprops=dict(arrowstyle='->'))
    util.plt.xlabel('x')
    util.plt.ylabel('f(x)')
    util.plt.show()

    # 鞍点，曲线的鞍点
    x = np.arange(-2.0, 2.0, 0.1)
    fig, = util.plt.plot(x, f2(x))
    fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
                      arrowprops=dict(arrowstyle='->'))
    util.plt.xlabel('x')
    util.plt.ylabel('f2(x)')
    util.plt.show()

    # 鞍点，曲面的鞍点
    x, y = np.mgrid[-1:1:31j, -1:1:31j]
    z = x ** 2 - y ** 2

    ax = util.plt.figure().add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    util.plt.xticks(ticks)
    util.plt.yticks(ticks)
    ax.set_zticks(ticks)
    util.plt.xlabel('x')
    util.plt.ylabel('y')
    util.plt.show()
