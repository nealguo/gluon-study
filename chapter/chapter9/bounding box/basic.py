from mxnet import image
from utils import util


# 定义边界框
def bbox_to_rect(bbox, color):
    return util.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


if __name__ == '__main__':
    # 显示图像
    util.set_figsize()
    img = image.imread('../img/catdog.jpg').asnumpy()
    util.plt.imshow(img)
    util.plt.show()

    # 显示边框
    dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
    fig = util.plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    util.plt.show()
