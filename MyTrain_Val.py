import os
import torch
import torch.nn.functional as F
from datetime import datetime
from torchvision.utils import make_grid
from lib.FCCNet import FCCNet
from utils_2.data_val import get_loader, test_dataset
from utils_2.utils import clip_gradient, adjust_lr, AvgMeter
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import logging
import numpy as np

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def train(train_loader, model, optimizer, epoch, writer):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_recordg, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    loss_all = 0
    epoch_step = 0
    save_path = '{}{}/'.format(opt.save_path, opt.train_save)
    # save_path = opt.train_save
    try:
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts, edges = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                edges = Variable(edges).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                lateral_map_g, lateral_map_3, lateral_map_2, lateral_map_1, edge_map = model(images)
                # ---- loss function ----
                lossg = structure_loss(lateral_map_g, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss1 = structure_loss(lateral_map_1, gts)
                # losse = dice_loss(edge_map, edges)
                losse = ce_loss(edge_map, edges)
                losse_lamada = opt.lamada * losse
                loss = loss1 + loss2 + loss3 + lossg + losse_lamada  # TODO: try different weights for loss
                loss_all += loss
                epoch_step += 1
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_recordg.update(lossg.data, opt.batchsize)
                    loss_recorde.update(losse_lamada.data, opt.batchsize)
            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-1: {:.4f}, lateral-2: {:0.4f}, lateral-3: {:0.4f}, lateral-g: {:0.4f}], lateral-e: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_recordg.show(),
                             loss_recorde.show()))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], lateral-1: {:.4f}, lateral-2: {:0.4f}, '
                    'lateral-3: {:0.4f}, lateral-g: {:0.4f}], lateral-e: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss_record1.show(), loss_record2.show(),
                           loss_record3.show(), loss_recordg.show(), loss_recorde.show()))
        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        os.makedirs(save_path, exist_ok=True)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_epoch, best_dice
    model.eval()
    with torch.no_grad():
        DSC_sum = 0.0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.upsample(res[3], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC_sum += dice
        Dsc = DSC_sum / test_loader.size
        writer.add_scalar('Dsc', torch.tensor(Dsc), global_step=epoch)

        if epoch == 1:
            best_dice = Dsc
        else:
            if Dsc > best_dice:
                best_dice = Dsc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'val_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        print('Epoch: {}, Dsc: {}, ValbestEpoch: {},bestDsc: {} '.format(epoch, Dsc, best_epoch, best_dice))
        logging.info(
            '[Val Info]:Epoch:{} Dsc:{} ValbestEpoch:{} bestDsc:{}'.format(epoch, Dsc, best_epoch, best_dice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_url', type=str, default='./pvt_v2_b2.pth',
                        help='model_name')  # TODO
    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5 * 1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./TrainDataset', help='path to train dataset')
    parser.add_argument('--val_root', type=str,
                        default='./ValDataset/', help='path to val dataset')
    parser.add_argument('--test_root', type=str,
                        default='./TestDataset/', help='path to val dataset')
    parser.add_argument('--train_save', type=str,
                        default='FCCNet')
    parser.add_argument('--save_path', type=str,
                        default='./save/',
                        help='the path to save model and log')
    parser.add_argument('--lamada', type=float,
                        default=3.0,
                        help='lamada*losse')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = FCCNet(channel=32).cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    val_loader = test_dataset(image_root=opt.val_root + 'images/',
                              gt_root=opt.val_root + 'masks/',
                              testsize=opt.trainsize)
    test_loader = test_dataset(image_root=opt.test_root + 'images/',
                               gt_root=opt.test_root + 'masks/',
                               testsize=opt.trainsize)
    total_step = len(train_loader)
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, save_path, opt.decay_epoch))

    writer = SummaryWriter(save_path + 'summary')
    best_dice = 0
    best_epoch = 0
    best_test_epoch = 0
    best_test_dice = 0
    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, writer)
        val(val_loader, model, epoch, save_path, writer)