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


import os
from PIL import Image
import matplotlib
import matplotlib.cm
import numpy as np
import paddle


class AverageMeter(object):
    '''
    用于保存训练时的loss值
    '''
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def DepthNorm(depth, maxDepth=1000.0):
    '''
    深度值标准化
    '''
    return maxDepth / depth


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    '''
    预测结果上色
    '''
    value = value.cpu().numpy()[0, :, :]
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2, 0, 1))


def load_dygraph_pretrain(model, path=None):
    '''
    DenseDepth Encoder 预训练权重加载
    '''
    if not (os.path.isdir(path) or os.path.exists(path)):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path)
    model.set_dict(param_state_dict)


def load_images(image_file):
    '''
    加载图像，水平翻转，返回tensor
    '''
    x = np.clip(np.asarray(Image.open(image_file).resize((640, 480)), dtype=float) / 255, 0, 1).transpose(2, 0, 1)

    return np.expand_dims(x, axis=0).astype(np.float32)