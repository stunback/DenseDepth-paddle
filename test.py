import time
import argparse
import datetime
import os

import paddle
import paddle.nn
import paddle.nn.utils
import cv2
import numpy as np

from model import DensDepthModel
from data import getTrainingTestingDataset
from utils import AverageMeter, DepthNorm, colorize, load_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, default='./weights/DenseDepth_best.pdparams',
                        help="path to checkpoint")
    parser.add_argument("--data", type=str, default="examples/", help="Path to dataset zip file")
    parser.add_argument("--cmap", type=str, default="gray", help="Colormap to be used for the predictions")
    args = parser.parse_args()
    return args


def main(args):
    if len(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("{} no such file".format(args.checkpoint))
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
        img = paddle.to_tensor(img)
        pred = model(img)

        out = DepthNorm(pred.squeeze(0))
        output = colorize(out, cmap=args.cmap)
        cv2.imwrite(savepath, np.transpose(output, (1, 2, 0)))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
