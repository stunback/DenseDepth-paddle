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


import argparse
import configparser
import os

import paddle
import paddle.nn
import paddle.nn.utils
import cv2
import numpy as np

from model.model import DensDepthModel
from utils.utils import DepthNorm, colorize, load_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, default='./logs/DenseDepth_val_best.pdparams',
                        help="path to checkpoint")
    parser.add_argument("--data", type=str, default="images/", help="Path to dataset zip file")
    parser.add_argument("--cmap", type=str, default="gray", help="Colormap to be used for the predictions")
    args = parser.parse_args()
    return args


def main(args):
    if len(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("{} no such file".format(args.checkpoint))
    # 加载模型
    model = DensDepthModel()
    model_state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(model_state_dict)
    model.eval()
    print('Model load success!')
    imgfiles = os.listdir(args.data)

    if not os.path.exists('./results'):
        os.mkdir('./results')

    for filename in imgfiles:
        filepath = os.path.join(args.data, filename)
        savepath = os.path.join('./results', filename)
        print("processing image {}".format(filepath))
        img = load_images(filepath)
        # 图片转tensor
        img = paddle.to_tensor(img)
        # 预测深度
        pred = model(img)
        # 深度标准化
        out = DepthNorm(pred.squeeze(0))
        # 预测结果上色
        output = colorize(out, cmap=args.cmap)
        # 保存结果
        cv2.imwrite(savepath, np.transpose(output, (1, 2, 0)))


if __name__ == '__main__':
    args = parse_arguments()
    cfg = configparser.ConfigParser()
    cfg.read('configs/main.cfg')
    args.checkpoint = cfg.get('predict', 'weights_path')
    args.data = cfg.get('predict', 'imagedir_path')
    args.cmap = cfg.get('predict', 'color_map')

    main(args)
