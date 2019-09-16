from mxnet import gluon, image
from utils import util


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = util.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


# 对输入的图像进行多次图像增广
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    util.plt.show()


if __name__ == '__main__':
    # 显示图像
    util.set_figsize()
    img = image.imread("../img/cat1.jpg")
    util.plt.imshow(img.asnumpy())
    util.plt.show()

    # 左右翻转
    flip_aug = gluon.data.vision.transforms.RandomFlipLeftRight()
    apply(img, flip_aug)
    # 上下翻转
    apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
    # 随机裁剪
    shape_aug = gluon.data.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)
    # 降低亮度
    apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
    # 调整色调
    apply(img, gluon.data.vision.transforms.RandomHue(0.5))
    # 调整亮度brightness、对比度contrast、饱和度saturation和色调hue
    color_aug = gluon.data.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug)
    # 叠加多种图像增广方法
    gluon.data.vision.transforms.RandomFlipLeftRight()
    apply(img, gluon.data.vision.transforms.Compose([flip_aug, shape_aug, color_aug]))
