from mxnet import gluon, init, nd
from mxnet.gluon import model_zoo
from utils import util
import os
import zipfile


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_imgs.transform_first(test_augs), batch_size)
    ctx = util.try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
    util.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


if __name__ == '__main__':
    # 下载数据集
    data_dir = '../data'
    base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
    fname = gluon.utils.download(
        base_url + 'gluon/dataset/hotdog.zip',
        path=data_dir,
        sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
    with zipfile.ZipFile(fname, 'r') as z:
        z.extractall(data_dir)

    # 加载数据集
    train_imgs = gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'hotdog/train'))
    test_imgs = gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'hotdog/test'))

    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    util.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    util.plt.show()

    # 指定RGB三个通道的均值和标准差来将图像通道归⼀化
    normalize = gluon.data.vision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.RandomResizedCrop(224),
        gluon.data.vision.transforms.RandomFlipLeftRight(),
        gluon.data.vision.transforms.ToTensor(), normalize])
    test_augs = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.Resize(256),
        gluon.data.vision.transforms.CenterCrop(224),
        gluon.data.vision.transforms.ToTensor(), normalize])

    # 定义模型，初始化模型
    # 制定pretrained=True来联网下载模型参数
    pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
    fintune_net = model_zoo.vision.resnet18_v2(classes=2)
    # features中的模型参数已经足够好，只需要较小的学习率来微调
    fintune_net.features = pretrained_net.features
    # output中的模型参数采用随机初始化，需要更大的学习率从头训练
    fintune_net.output.initialize(init=init.Xavier())
    # output中的模型参数将在迭代中使用10倍大的学习率
    fintune_net.output.collect_params().setattr('lr_mult', 10)

    # 训练并测试
    # 使用学习率为0.01的Trainer，以便微调预训练得到的模型参数
    train_fine_tuning(fintune_net, learning_rate=0.01)
    # 使用较大的学习率0.1从头训练，以便对比所有模型参数都为随机值的相同模型
    scratch_net = model_zoo.vision.resnet18_v2(classes=2)
    scratch_net.initialize(init=init.Xavier())
    train_fine_tuning(scratch_net, 0.1)
