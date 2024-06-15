

import os
import time
import math
import numpy as np
import torch
import torch.utils.data as data
from data import WiderCardDetection, preproc, detection_collate


def validate(net, cfg, dataset, device, criterion, priors):
    net.eval()

    print('dataset', len(dataset))
    batch_size = cfg['batch_size']
    print('batch_size', batch_size)
    epoch_size = math.ceil(len(dataset) / batch_size)
    print('epoch_size', epoch_size)

    validation_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate)

    loss_list = [[], [], [], []]
    elapse_list = []
    for iteration in range(0, epoch_size):
        # create batch iterator
        batch_iterator = iter(validation_loader)
        load_t0 = time.time()
        # load train data
        images, targets = next(batch_iterator)
        # if torch.cuda.is_available():
        #     images = images.cuda()
        #     targets = [anno.cuda() for anno in targets]
        # else:
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]
        # print(images)
        # print(type(images), images.shape, images.dtype)
        # forward
        out = net(images)
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        load_t1 = time.time()
        loss_list[0].append(loss_l.item())
        loss_list[1].append(loss_c.item())
        loss_list[2].append(loss_landm.item())
        loss_list[3].append(loss.item())
        elapse_list.append(load_t1 - load_t0)
    tmp = [np.average(l) for l in loss_list]
    return tmp, np.sum(elapse_list)

