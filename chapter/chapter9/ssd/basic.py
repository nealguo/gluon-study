from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import nn
from utils import util
import time


def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_preditor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    block.initialize()
    return block(x)


def flatten_pred(pred):
    return pred.transposse((0, 2, 3, 1)).flatten()


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


# 高和宽减半块
def down_sample_block(num_channels):
    block = nn.Sequential()
    for _ in range(2):
        block.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                  nn.BatchNorm(in_channels=num_channels),
                  nn.Activation('relu'))
    block.add(nn.MaxPool2D(2))
    return block


# 基础网络块
def base_net():
    block = nn.Sequential()
    for num_filters in [16, 32, 64]:
        block.add(down_sample_block(num_filters))
    return block


def get_block(i):
    if i == 0:
        block = base_net()
    elif i == 4:
        block = nn.GlobalMaxPool2D()
    else:
        block = down_sample_block(128)
    return block


def block_forwad(X, block, size, ratio, cla_predictor, bbox_predictor):
    Y = block(X)
    anchors = contrib.nd.MultiBoxPrior(Y, size=size, ratio=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 赋值语句 self.blk_i = get_block(i)
            setattr(self, 'blk_%d' % i, get_block(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i, bbox_preditor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d'% i) 即访问 self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = block_forwad(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                concat_preds(bbox_preds))


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()


def bbox_eval(bbox_preds, bbox_labeles, bbox_masks):
    return ((bbox_labeles - bbox_preds) * bbox_masks).abs().sum().asscalar()


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]


def display(img, output, threshold):
    fig = util.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        util.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        util.plt.show()


if __name__ == '__main__':
    Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
    Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
    print(Y1.shape)
    print(Y2.shape)

    # 连接多尺度的预测
    print(concat_preds([Y1, Y2]).shape)
    print(forward(nd.zeros((2, 3, 20, 20)), down_sample_block(10)).shape)
    print(forward(nd.zeros((2, 3, 256, 256)), base_net()).shape)

    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    # 创建模型
    net = TinySSD(num_classes=1)
    net.initialize()
    X = nd.zeros((32.3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

    # 读物数据集，初始化
    batch_size = 32
    train_iter, _ = util.load_data_pikachu(batch_size)
    ctx, net = util.try_gpu(), TinySSD(num_classes=1)
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})

    # 定义损失函数和评价函数
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    bbox_loss = gluon.loss.L1Loss()

    # 训练模型
    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # 从头读取数据
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # ⽣成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = net(X)
                # 为每个锚框标注类别和偏移量
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(anchors, Y,
                                                                                cls_preds.transpose((0, 2, 1)))
                # 根据类别和偏移量的预测和标注值计算损失函数
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            trainer.step(batch_size)
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))

    # 预测
    img = image.imread('../img/pikachu.jpg')
    feature = image.imresize(img, 256, 256).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
    output = predict(X)
    util.set_figsize((5, 5))
    display(img, output, threshold=0.3)
