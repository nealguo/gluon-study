from utils import util
import numpy as np


# 一维梯度下降
def gd(eta):
    x = 10
    results = [x]
    for i in range(x):
        x -= eta * 2 * x  # 一维目标函数f(x) = x * x
        results.append(x)
    print('epoch 10, x:', x)
    return results


# 显示一维轨迹
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    util.set_figsize()
    util.plt.plot(f_line, [x * x for x in f_line])  # 显示f(x)=x*x的图像
    util.plt.plot(res, [x * x for x in res], '-o')  # 显示梯度下降图像
    util.plt.xlabel('x')
    util.plt.ylabel('f(x)')
    util.plt.show()


# 使用二维梯度下降来训练
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (20, x1, x2))
    return results


# 显示二维轨迹
def show_trace_2d(f, results):
    util.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    util.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    util.plt.xlabel('x1')
    util.plt.ylabel('x2')
    util.plt.show()


# 二维目标函数
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


# 二维梯度下降
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


# 二维随机梯度下降（增加随机噪声来模拟随机梯度下降）
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)


if __name__ == '__main__':
    # 一维梯度下降
    # 学习率eta=0.2, 学习率合适，在x=0.06处已经效果明显
    res = gd(0.2)
    show_trace(res)

    # 学习率eta=0.05, 学习率过小，梯度下降较慢
    res = gd(0.05)
    show_trace(res)

    # 学习率eta=1.1, 学习率过大，左右摇摆
    res = gd(1.1)
    show_trace(res)

    # 二维梯度下降
    eta = 0.1
    f = f_2d
    trainer = gd_2d
    res2 = train_2d(trainer)
    show_trace_2d(f, res2)

    # 二维随机梯度下降
    show_trace_2d(f_2d, train_2d(sgd_2d))
