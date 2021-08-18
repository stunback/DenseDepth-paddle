import time
import argparse
import datetime

import paddle
import paddle.nn
import paddle.nn.utils
import visualdl

from tensorboardX import SummaryWriter
from model import DensDepthModel
from losses import all_loss
from data import getTrainingTestingDataset
from utils import AverageMeter, DepthNorm, colorize


def parse_arguments():
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()
    return args


def Logging(model, writer, val_loader, epoch):
    model.eval()
    for batch_id, data in enumerate(val_loader):
        depth_pred = model(data['image'])
        depth_true = data['depth']


def main(args):
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    model = DensDepthModel()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)

    traindataset, valdataset = getTrainingTestingDataset()
    train_loader = paddle.io.DataLoader(traindataset, batch_size=bs)
    val_loader = paddle.io.DataLoader(valdataset, batch_size=bs)

    prefix = 'densenet_' + str(bs)
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    paddle.set_device('gpu')
    for epoch in range(epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        val_losses = AverageMeter()
        N = len(train_loader)
        min_val_loss = 10

        end = time.time()
        model.train()
        for i, sampled_batch in enumerate(train_loader):
            depth_pred = model(sampled_batch['image'])
            depth_gt = DepthNorm(sampled_batch['depth'])
            loss = all_loss(depth_gt, depth_pred, theta=1.0)
            losses.update(float(loss[0]), sampled_batch['image'].shape[0])

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            niter = epoch * N + i
            if i % 10 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {lossval:.4f} ({lossavg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, lossval=losses.val, lossavg=losses.avg, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

            if i % 2400 == 0 :
                model.eval()
                val_start = time.time()
                for i, sampled_batch in enumerate(val_loader):
                    val_depth_pred = model(sampled_batch['image'])
                    val_depth_gt = DepthNorm(sampled_batch['depth'])
                    val_loss = all_loss(val_depth_gt, val_depth_pred, theta=1.0)
                    val_losses.update(float(val_loss[0]), sampled_batch['image'].shape[0])

                if val_losses.avg < min_val_loss and epoch > 0:
                    min_val_loss = val_losses.avg
                    paddle.save(model.state_dict(), "./logs2/DenseDepth_val_best_{}.pdparams".format(epoch))
                    paddle.save(optimizer.state_dict(), "./logs2/Adam_val_best_{}.pdopt".format(epoch))

                val_end = time.time()
                print('Val Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, N),
                      'Val Time: %.3f \t' % (val_end - val_start),
                      'Val Loss: %.3f ' % (val_losses.avg))
                val_losses.reset()
                model.train()
        # save
        paddle.save(model.state_dict(), "./logs2/DenseDepth_epochs_{}.pdparams".format(epoch))
        paddle.save(optimizer.state_dict(), "./logs2/Adam_epochs_{}.pdopt".format(epoch))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
