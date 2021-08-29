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

import paddle
import paddle.nn
import paddle.nn.utils

from model.model import DensDepthModel
from data.data import getTestingDataset
from utils.utils import DepthNorm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, default='logs/DenseDepth_best.pdparams',
                        help="path to checkpoint")
    args = parser.parse_args()
    return args


def compute_errors(gt, pred):
    thresh = paddle.maximum((gt / pred), (pred / gt)).numpy()
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = paddle.mean(paddle.abs(gt - pred) / gt).numpy()
    rmse = (gt - pred) ** 2
    rmse = paddle.sqrt(rmse.mean()).numpy()
    log_10 = (paddle.abs(paddle.log10(gt)-paddle.log10(pred))).mean().numpy()
    return a1, a2, a3, abs_rel, rmse, log_10


def main(args):
    model = DensDepthModel()
    model_state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(model_state_dict)
    model.eval()

    testset = getTestingDataset()

    preds = paddle.zeros([len(testset), 240, 320], dtype=paddle.float32)
    gts = paddle.zeros([len(testset), 240, 320], dtype=paddle.float32)
    paddle.set_device('gpu')
    for i in range(len(testset)):
        print(i)
        depth_pred = DepthNorm(model(paddle.unsqueeze(testset[i]['image'], axis=0)))[0][0]
        depth_pred_flip = DepthNorm(model(paddle.unsqueeze(testset[i]['image'][:, :, ::-1], axis=0)))[0][0]
        depth_gt = testset[i]['depth'][0]
        depth_pred_merge = depth_pred * 0.5 + paddle.flip(depth_pred_flip, axis=-1) * 0.5
        del depth_pred
        del depth_pred_flip
        # Scaled
        scaled = paddle.mean(depth_gt) / paddle.mean(depth_pred_merge)
        preds[i] = depth_pred_merge * scaled
        gts[i] = depth_gt
        # My GPU Memory is not enough
        del scaled
        del depth_gt
        del depth_pred_merge

    predictions = paddle.clip(preds, 40, 1000) / 100.
    groundtruths = paddle.clip(gts, 40, 1000) / 100.
    # Eigen crop
    e = compute_errors(groundtruths[:, 12:225, 15:310], predictions[:, 12:225, 15:310])
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(e[0], e[1], e[2], e[3][0], e[4][0], e[5][0]))


if __name__ == '__main__':
    args = parse_arguments()
    cfg = configparser.ConfigParser()
    cfg.read('configs/main.cfg')
    args.checkpoint = cfg.get('eval', 'weights_path')

    main(args)
