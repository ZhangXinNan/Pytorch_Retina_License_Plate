from __future__ import print_function
import os

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderLPDetection, detection_collate, preproc
from data import WiderCardDetection
from data import cfg_mnet_zx as cfg_mnet
from data import cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retina import Retina
from draw_chart import draw, draw2
from val import validate


def set_device(gpu_id=0):
    num_gpu = 0
    gpu_train = False
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        num_gpu = 1
        gpu_train = True
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(device)
    return device, num_gpu, gpu_train


def get_args():
    parser = argparse.ArgumentParser(description='RetinaLP Training')
    parser.add_argument('--training_dataset',
                        # default='/home/xin.zhang6/data_br/val.txt',
                        default='D:\\data_br\\MexicoVotar_20240524_2222_train_val\\val.txt',
                        help='Training dataset directory')
    parser.add_argument('--validation_dataset',
                        # default='/home/xin.zhang6/data_br/test.txt',
                        default='D:\\data_br\\MexicoVotar_20240524_2222_train_val\\test.txt'
                        )
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net',
                        default='weights/mobilenet0.25_epoch_8_white_ccpd.pth',
                        help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights_card_20240614_512_320/', help='Location to save checkpoint models')
    parser.add_argument('--batch_size', default=4, type=int)
    return parser.parse_args()


def train():
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    # dataset = WiderLPDetection(training_dataset, preproc(img_dim, rgb_mean))
    dataset = WiderCardDetection(training_dataset, preproc(img_dim, rgb_mean))
    print('dataset', len(dataset))
    print('batch_size', batch_size)
    epoch_size = math.ceil(len(dataset) / batch_size)
    iteration = 0
    max_iter = max_epoch * epoch_size
    print('epoch_size', epoch_size)
    # print('max_iter', max_iter)

    # stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    # step_index = 0

    # if args.resume_epoch > 0:
    #     start_iter = args.resume_epoch * epoch_size
    # else:
    #     start_iter = 0
    epoch_list = []
    loss_list = [[], [], [], []]
    val_loss_list = [[], [], [], []]
    min_val_loss = 99999999
    fout = open(os.path.join(save_folder, 'loss_chart.txt'), 'w')

    # for iteration in range(start_iter, max_iter):
        # if iteration % epoch_size == 0:
    for epoch in range(1, max_epoch + 1):
        training_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
        loss_tmp = [[], [], [], []]
        lr = args.lr
        epoch_time0 = time.time()
        net.train()
        for iter, (images, targets) in enumerate(training_loader):
            load_t0 = time.time()
            iteration += 1
            # if iteration in stepvalues:
            #     step_index += 1
            # lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
            # load train data
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]
            # print(images)
            # print(type(images), images.shape, images.dtype)
            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            # print(out)
            # print(priors)
            # print(targets)
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            # 保存损失到列表里
            loss_tmp[0].append(loss_l.item())
            loss_tmp[1].append(loss_c.item())
            loss_tmp[2].append(loss_landm.item())
            loss_tmp[3].append(loss.item())

            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s'
                .format(epoch, max_epoch, iter, epoch_size, iteration, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time))
        epoch_time = time.time() - epoch_time0
        # 每个epoch跑一遍验证集
        val_loss_epoch, elapse_epoch = validate(net, cfg, args.validation_dataset, device, criterion, priors)
        # 打印损失列表
        epoch_list.append(epoch)
        for i in range(4):
            loss_list[i].append(np.average(loss_tmp[i]))
        for i in range(4):
            val_loss_list[i].append(val_loss_epoch[i])
        print(epoch, f"train-loss: {[l[-1] for l in loss_list]}", f"val-loss: {val_loss_epoch}", f"train-time:{epoch_time}, val-time:{elapse_epoch}")
        # 只保存最佳模型（损失最小）
        if val_loss_epoch[-1] < min_val_loss:
            min_val_loss = val_loss_epoch[-1]
            torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '_card.pth')
        # 画loss变换曲线，并输出到txt里
        # draw(epoch_list, loss_list, ['l', 'c', 'landm'], os.path.join(save_folder, f'loss_chart.{epoch}.png'))
        # draw(epoch_list, val_loss_list, ['l', 'c', 'landm'], os.path.join(save_folder, f'loss_chart.{epoch}.val.png'))
        draw2(epoch_list, [loss_list[0], val_loss_list[0]], ['l-train', 'l-val'], os.path.join(save_folder, f'loss_chart.{epoch}.l.png'))
        draw2(epoch_list, [loss_list[1], val_loss_list[1]], ['c-train', 'c-val'], os.path.join(save_folder, f'loss_chart.{epoch}.c.png'))
        draw2(epoch_list, [loss_list[2], val_loss_list[2]], ['landm-train', 'landm-val'], os.path.join(save_folder, f'loss_chart.{epoch}.landm.png'))
        draw2(epoch_list, [loss_list[3], val_loss_list[3]], ['loss-train', 'loss-val'], os.path.join(save_folder, f'loss_chart.{epoch}.loss.png'))
        if len(epoch_list) >= 10:
            draw(epoch_list[-10:], [l[-10:] for l in loss_list], ['l', 'c', 'landm'], os.path.join(save_folder, f'loss_chart.{epoch}.last10.png'))
            draw(epoch_list[-10:], [l[-10:] for l in val_loss_list], ['l', 'c', 'landm'], os.path.join(save_folder, f'loss_chart.{epoch}.val.last10.png'))
        line = f"{epoch}"
        for i in range(4):
            line += f" {loss_list[i][-1]}"
        for i in range(4):
            line += f" {val_loss_list[i][-1]}"
        fout.write(line + "\n")
        fout.flush()

    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')
    fout.close()


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def resume_model(net, resume_net, device):
    print('Loading resume network...')
    state_dict = torch.load(resume_net, map_location=device)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


# def main(args):
if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    device, num_gpu, gpu_train = set_device(cfg['gpu_id'])

    # rgb_mean = (104, 117, 123)  # bgr order
    rgb_mean = cfg['rgb_mean']
    num_classes = 2
    img_dim = cfg['image_size']
    # num_gpu = cfg['ngpu']
    # batch_size = cfg['batch_size']
    cfg['batch_size'] = batch_size = args.batch_size
    max_epoch = cfg['epoch']
    # gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder

    net = Retina(cfg=cfg)
    print("Printing net...")

    if args.resume_net is not None:
        net = resume_model(net, args.resume_net, device)

    # if num_gpu > 1 and gpu_train:
    #     net = torch.nn.DataParallel(net).cuda()
    # else:
    net = net.to(device)
    # elif torch.cuda.is_available():
    #     net = net.cuda()
    cudnn.benchmark = True
    # cudnn.benchmark = False

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    criterion.set_device(device)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        # if torch.cuda.is_available():
        #     priors = priors.cuda()
    priors = priors.to(device)
    train()
# if __name__ == '__main__':
    # train()
    # main(get_args())