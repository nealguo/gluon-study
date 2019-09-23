from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
from utils import util
import time


def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)


def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)


def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, ctx):
    content_X = preprocess(content_img, image_shape).copyto(ctx)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, ctx):
    style_X = preprocess(style_img, image_shape).copyto(ctx)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    return (Y_hat - Y).square().mean()


def gram(X):
    num_channels, n = X.shape[1], X.size // X.shape[1]
    X = X.reshape((num_channels, n))
    return nd.dot(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return (gram(Y_hat) - gram_Y).square().mean()


def tv_loss(Y_hat):
    return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() + (
            Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = nd.add_n(*styles_l) + nd.add_n(*contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self): return self.weight.data()


def get_inits(X, ctx, lr, styles_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam', {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


def train(X, contents_Y, styles_Y, ctx, lr, max_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, ctx, lr, styles_Y)
    for i in range(max_epochs):
        start = time.time()
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        nd.waitall()
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, ' 'TV loss %.2f, %.2f sec' % (
                i, nd.add_n(*contents_l).asscalar(), nd.add_n(*styles_l).asscalar(), tv_l.asscalar(),
                time.time() - start))
        if i % lr_decay_epoch == 0 and i != 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            print('change lr to %.1e' % trainer.learning_rate)
    return X


if __name__ == '__main__':
    util.set_figsize()
    content_img = image.imread('../img/rainier.jpg')
    util.plt.imshow(content_img.asnumpy())

    style_img = image.imread('../img/autumn_oak.jpg')
    util.plt.imshow(style_img.asnumpy())
    util.plt.show()

    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])

    pretrained_net = model_zoo.vision.vgg19(pretrained=True)
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    net = nn.Sequential()
    for i in range(max(content_layers + style_layers) + 1):
        net.add(pretrained_net.features[i])

    content_weight, style_weight, tv_weight = 1, 1e3, 10
    ctx, image_shape = util.try_gpu(), (225, 150)
    net.collect_params().reset_ctx(ctx)
    content_X, contents_Y = get_contents(image_shape, ctx)
    _, styles_Y = get_styles(image_shape, ctx)

    output = train(content_X, contents_Y, styles_Y, ctx, 0.01, 500, 200)
    util.plt.imsave('../img/neural-style-1.png', postprocess(output).asnumpy())
    util.plt.show()

    image_shape = (450, 300)
    _, content_Y = get_contents(image_shape, ctx)
    _, style_Y = get_styles(image_shape, ctx)
    X = preprocess(postprocess(output) * 255, image_shape)
    output = train(X, content_Y, style_Y, ctx, 0.01, 300, 100)
    util.plt.imsave('../img/neural-style-2.png', postprocess(output).asnumpy())
    util.plt.show()
