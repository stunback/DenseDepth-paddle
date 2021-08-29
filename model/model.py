#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from model.densenet import DenseNet169


class Encoder(paddle.nn.Layer):
    '''
    DenseDepth编码网络部分，采用DenseNet169
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.pretrained_model = DenseNet169(pretrained='./model/DenseNet169_pretrained.pdparams')

    def forward(self, x):
        y, features = self.pretrained_model(x)
        return features


class Decoder(paddle.nn.Layer):
    '''
    DenseDepth解码网络
    '''
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
    '''
    上采样，采用双线性插值+卷积
    '''
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
    '''
    DenseNet169 Encoder + Bilinear interpolation Decoder
    '''
    def __init__(self):
        super(DensDepthModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

