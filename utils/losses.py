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
import paddle.nn.functional as F


def ssim(img1, img2, L=100.):
    """Calculate SSIM (structural similarity) for one channel images.
    计算depth predict和depth gt的SSIM
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
    Returns:
        float: ssim result.
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    C3 = C2/2

    # ux
    ux = paddle.mean(img1)
    # uy
    uy = paddle.mean(img2)
    # ux^2
    ux_sq = ux**2
    # uy^2
    uy_sq = uy**2
    # ux*uy
    uxuy = ux * uy
    # ox、oy方差计算
    ox_sq = paddle.var(img1)
    oy_sq = paddle.var(img2)
    ox = paddle.sqrt(ox_sq)
    oy = paddle.sqrt(oy_sq)
    oxoy = ox * oy
    oxy = paddle.mean((img1 - ux) * (img2 - uy))

    L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
    C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
    S = (oxy + C3) / (oxoy + C3)
    ssim = L * C * S
    return ssim


def image_gradients(img):
    """
    计算图像的x和y方向梯度
    """
    if len(img.shape) != 4:
        raise ValueError("Shape mismatch. Needs to be 4 dim tensor")

    img_shape = img.shape
    batch_size, channels, height, width = img.shape
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]

    dy = paddle.concat([dy, paddle.zeros([batch_size, channels, 1, width], dtype=img.dtype)], axis=2)
    dy = paddle.reshape(dy, img_shape)

    dx = paddle.concat([dx, paddle.zeros([batch_size, channels, height, 1], dtype=img.dtype)], axis=3)
    dx = paddle.reshape(dx, img_shape)

    return dy, dx


def grad_loss(y_true, y_pred):
    '''
    计算梯度loss
    '''
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)
    l_edges = paddle.mean(paddle.abs(dy_pred - dy_true) + paddle.abs(dx_pred - dx_true), axis=1)
    return l_edges


def depth_loss(y_true, y_pred):
    '''
    计算深度值的L1loss
    '''
    l1loss = paddle.nn.L1Loss()
    output =l1loss(y_pred, y_true)
    return output


def all_loss(y_true, y_pred, theta=1.0, maxdepthval=1000./10.):
    '''
    计算SSIM、梯度和深度值L1loss的整体loss
    '''
    l1_loss = depth_loss(y_true, y_pred)
    ssim_loss = paddle.clip((1 - ssim(y_pred, y_true, maxdepthval)) * 0.5, min=0, max=1)
    gradient_loss = grad_loss(y_true, y_pred)
    loss = theta * l1_loss + 1.0 * ssim_loss + 1.0 * paddle.mean(gradient_loss)

    return loss
