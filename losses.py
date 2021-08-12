import math
import paddle
import paddle.nn.functional as F
import numpy as np
import cv2


def gaussian(window_size, sigma):
    gauss = paddle.Tensor(np.array([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]))
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand([channel, 1, window_size, window_size])
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    _, channel, height, width = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = paddle.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret


def ssim_simple(img1, img2, L=100.):
    """Calculate SSIM (structural similarity) for one channel images.
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
    # 公式一计算
    L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
    C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
    S = (oxy + C3) / (oxoy + C3)
    ssim = L * C * S
    # 验证结果输出
    # print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
    return ssim


def image_gradients(img):
    """works like tf one"""
    if len(img.shape) != 4:
        raise ValueError("Shape mismatch. Needs to be 4 dim tensor")

    img_shape = img.shape
    batch_size, channels, height, width = img.shape
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]

    shape = np.stack([batch_size, channels, 1, width])
    dy = paddle.concat([dy, paddle.zeros([batch_size, channels, 1, width], dtype=img.dtype)], axis=2)
    dy = paddle.reshape(dy, img_shape)

    shape = np.stack([batch_size, channels, height, 1])
    dx = paddle.concat([dx, paddle.zeros([batch_size, channels, height, 1], dtype=img.dtype)], axis=3)
    dx = paddle.reshape(dx, img_shape)

    return dy, dx


def grad_loss(y_true, y_pred):
    # Edges
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)
    l_edges = paddle.mean(paddle.abs(dy_pred - dy_true) + paddle.abs(dx_pred - dx_true), axis=1)
    return l_edges


def depth_loss(y_true, y_pred):
    l1loss = paddle.nn.L1Loss()
    output =l1loss(y_pred, y_true)
    return output


def all_loss(y_true, y_pred, theta=1.0, maxdepthval=1000./10.):
    l1_loss = depth_loss(y_true, y_pred)
    ssim_loss = paddle.clip((1 - ssim_simple(y_pred, y_true, maxdepthval)) * 0.5, min=0, max=1)
    gradient_loss = grad_loss(y_true, y_pred)
    loss = theta * l1_loss + 1.0 * ssim_loss + 1.0 * paddle.mean(gradient_loss)

    return loss
