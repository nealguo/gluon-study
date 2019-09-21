from mxnet import contrib, gluon, image, nd
from utils import util
import numpy as np


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = util.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


if __name__ == '__main__':
    # 修改Numpy的打印精度
    np.set_printoptions(2)

    img = image.imread('../img/catdog.jpg').asnumpy()
    h, w = img.shape[0:2]
    print(h, w)
    # 生成锚框变量Y的形状（批量大小，锚框个数，4）
    X = nd.random.uniform(shape=(1, 3, h, w))
    Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)

    boxes = Y.reshape((h, w, 5, 4))
    print(boxes[250, 250, 0, :])

    util.set_figsize()
    bbox_scale = nd.array((w, h, w, h))
    fig = util.plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75,r=1', 's=0.5,r=1', 's=0.25,r=1', 's=0.75,r=2', 's=0.75,r=0.5'])
    util.plt.show()

    ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])

    fig = util.plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    util.plt.show()

    labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0), ground_truth.expand_dims(axis=0),
                                       nd.zeros((1, 3, 5)))
    anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = nd.array([0] * anchors.size)
    cls_probs = nd.array([[0] * 4,  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    fig = util.plt.imshow(img)
    show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    util.plt.show()

    output = contrib.nd.MultiBoxDetection(cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0),
                                          anchors.expand_dims(axis=0), nms_threshold=0.5)
    print(output)
    fig = util.plt.imshow(img)
    for i in output[0].asnumpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)
    util.plt.show()
