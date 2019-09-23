from mxnet import gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
from utils import util
import numpy as np
import sys


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmax(net(X.as_in_context(ctx[0])), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(pred):
    colormap = nd.array(util.VOC_COLORMAP, ctx=ctx[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]


if __name__ == '__main__':
    X = nd.arange(1, 17).reshape((1, 1, 4, 4))
    K = nd.arange(1, 10).reshape((1, 1, 3, 3))
    conv = nn.Conv2D(channels=1, kernel_size=3)
    conv.initialize(init.Constant(K))
    print(conv(X), K)

    W, k = nd.zeros((4, 16)), nd.zeros(11)
    k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
    W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
    print(nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)), W)

    conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
    conv.initialize()
    X = nd.random.uniform(shape=(1, 3, 64, 64))
    Y = conv(X)
    print(Y.shape)

    conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
    conv_trans.initialize()
    print(conv_trans(Y).shape)

    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
    print(pretrained_net.features[-4:], pretrained_net.output)

    net = nn.HybridSequential()
    for layer in pretrained_net.features[:-2]:
        net.add(layer)
    X = nd.random.uniform(shape=(1, 3, 320, 480))
    print(net(X).shape)

    num_classes = 21
    net.add(nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))

    conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
    conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))

    img = image.imread('../img/catdog.jpg')
    X = img.astype('float32').transpose((2, 0, 1)).expand_dims(axis=0) / 255
    Y = conv_trans(X)
    out_img = Y[0].transpose((1, 2, 0))

    util.set_figsize()
    print('input image shape:', img.shape)
    util.plt.imshow(img.asnumpy())
    print('output image shape:', out_img.shape)
    util.plt.imshow(out_img.asnumpy())

    net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)))
    net[-2].initialize(init=init.Xavier())

    crop_size, batch_size, colormap2label = (320, 480), 32, nd.zeros(256 ** 3)
    for i, cm in enumerate(util.VOC_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
    voc_dir = util.download_voc_pascal(data_dir='../data')
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gluon.data.DataLoader(util.VOCSegDataset(True, crop_size, voc_dir, colormap2label), batch_size,
                                       shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(util.VOCSegDataset(False, crop_size, voc_dir, colormap2label), batch_size,
                                      last_batch='discard', num_workers=num_workers)

    ctx = util.try_all_gpus()
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 1e-3})
    util.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=5)

    test_images, test_labels = util.read_voc_images(is_train=False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 480, 320)
        X = image.fixed_crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
    util.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
