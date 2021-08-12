import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from densenet import DenseNet169


class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pretrained_model = DenseNet169(pretrained='./weights/DenseNet169_pretrained.pdparams')

    def forward(self, x):
        y, features = self.pretrained_model(x)
        return features


class Decoder(paddle.nn.Layer):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2D(num_features, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features//1 + 1280, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 512,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 256,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)
        self.conv3 = nn.Conv2D(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class UpSample(paddle.nn.Layer):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()

        self.convA = nn.Conv2D(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2D(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.shape[-2], concat_with.shape[-1]], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(paddle.concat([up_x, concat_with], axis=1)))))


class DensDepthModel(paddle.nn.Layer):
    def __init__(self):
        super(DensDepthModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    # a = Encoder()
    inp = paddle.to_tensor(np.ones([1, 3, 224, 224], dtype=np.float32))
    # outs = a(inp)
    # for out in outs:
    #    print(out.shape)

    b = DensDepthModel()
    outs = b(inp)
    print(outs.shape)
