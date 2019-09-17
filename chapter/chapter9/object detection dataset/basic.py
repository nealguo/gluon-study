from mxnet import gluon, image
from utils import util
import os


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/' 'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gluon.utils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)


def load_data_pikachu(batch_size, edge_size=256):  # edge_size：输出图像的宽和⾼
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir, 'train.rec'),
                                    path_imgidx=os.path.join(data_dir, 'train.idx'),
                                    batch_size=batch_size,
                                    data_shape=(3, edge_size, edge_size),  # 输出图像的形状
                                    shuffle=True,  # 以随机顺序读取数据集
                                    rand_crop=1,  # 随机裁剪的概率为1
                                    min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir, 'val.rec'),
                                  batch_size=batch_size,
                                  data_shape=(3, edge_size, edge_size),
                                  shuffle=False)
    return train_iter, val_iter


if __name__ == '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_pikachu(batch_size, edge_size)
    batch = train_iter.next()
    print(batch.data[0].shape, batch.label[0].shape)

    imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
    axes = util.show_images(imgs, 2, 5).flatten()
    for ax, label in zip(axes, batch.label[0][0:10]):
        util.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        util.plt.show()
