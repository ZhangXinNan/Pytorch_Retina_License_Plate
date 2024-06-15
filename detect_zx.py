from __future__ import print_function
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
# from data import cfg_mnet, cfg_re50
from data import cfg_mnet_zx as cfg_mnet, cfg_re50_zx as cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retina import Retina
from utils.box_utils import decode, decode_landm
import time
import torchvision
print(torch.__version__, torchvision.__version__)


CARD_WIDTH = 1024
CARD_HEIGHT = 640


def get_args():
    parser = argparse.ArgumentParser(description='RetinaPL')
    # 23 good
    parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_epoch_20_ccpd.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    # parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    parser.add_argument('-image', default='test_images/0.jpg', help='test image path')
    parser.add_argument('--out_dir', default='test_images_result/512-320')
    args = parser.parse_args()
    return args


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = Retina(cfg=cfg, phase='test')
    # net = load_model(net, args.trained_model, args.cpu)
    net = load_model(net, args.trained_model, not torch.cuda.is_available())
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1
    img_path_list = []
    if os.path.isfile(args.image):
        img_path_list.append(args.image)
    elif os.path.isdir(args.image):
        for filename in os.listdir(args.image):
            name, suffix = os.path.splitext(filename)
            if suffix.lower() not in ['.jpeg', '.png', '.jpg']:
                continue
            img_path_list.append(os.path.join(args.image, filename))
    # testing begin
    for img_path in img_path_list:
        
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = img_raw.shape[:2]
        resize = cfg['image_size'] / min(h, w)
        new_h, new_w = int(h * resize), int(w * resize)
        img_raw = cv2.resize(img_raw, (new_w, new_h))
        '''
        resize = cfg['image_size'] / max(h, w)
        new_h, new_w = int(h * resize), int(w * resize)
        img_raw = cv2.resize(img_raw, (new_w, new_h))
        img_square = np.zeros((max(new_w, new_h), max(new_w, new_h), 3), np.uint8)
        img_square[:new_h, :new_w, :] = img_raw
        img_raw = img_square
        cv2.imwrite("img_square.jpg", img_raw)
        '''
        # img_raw = cv2.resize(img_raw, (cfg['image_size'], cfg['image_size']))
        # resize_wh = (cfg['image_size'] / w, cfg['image_size'] / h)
        # print("resize_wh : ", resize_wh)
        img = np.float32(img_raw)
        print(img_path, img.shape, img.dtype)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        print(img.shape)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        print(type(loc), type(conf), type(landms))
        print(loc.shape, conf.shape, landms.shape)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        print(type(priors), priors.shape, type(priors.data), priors.data.shape)
        print("priors:", priors)
        prior_data = priors.data
        print("prior_data:", prior_data)
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        # boxes = boxes * scale / resize
        # resize = torch.Tensor([resize_wh[0], resize_wh[1], resize_wh[0], resize_wh[1]])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        print("boxes : ", boxes)
        # boxes[:, 0] /= resize_wh[0]
        # boxes[:, 1] /= resize_wh[1]
        # boxes[:, 2] /= resize_wh[0]
        # boxes[:, 3] /= resize_wh[1]
        # print("boxes : ", boxes)
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        # resize = torch.Tensor([resize_wh[0], resize_wh[1], resize_wh[0], resize_wh[1],
        #                        resize_wh[0], resize_wh[1], resize_wh[0], resize_wh[1]])
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        print("landms : ", landms)
        # for i in range(4):
        #     landms[2 * i] /= resize_wh[0]
        #     landms[2 * i + 1] /= resize_wh[1]
        # print("landms : ", landms)

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print('priorBox time: {:.4f}'.format(time.time() - tic))
        # show image
        if args.save_image:
            img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w = img_raw.shape[:2]
            for j, b in enumerate(dets):
                print(b)
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                print(text)
                b = list(map(int, b))
                for i in [0, 2, 5, 7, 9, 11]:
                    if b[i] < 0:
                        b[i] = 0
                    elif b[i] >= w:
                        b[i] = w - 1
                for i in [1, 3, 6, 8, 10, 12]:
                    if b[i] < 0:
                        b[i] = 0
                    elif b[i] >= h:
                        b[i] = h - 1
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 5)

                cv2.putText(img_raw, text, (b[0], b[1] + 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # landms
                cv2.circle(img_raw, (b[5], b[6]), 3, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 3, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 3, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[11], b[12]), 3, (255, 0, 0), 4)
                
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                w = int(x2 - x1 + 1.0)
                h = int(y2 - y1 + 1.0)
                img_box = np.zeros((h, w, 3))
                img_box = img_raw[y1:y2 + 1, x1:x2 + 1, :]
                # cv2.imshow("img_box",img_box)  
                # print('+++',b[9],b[10])
                
                # new_x1, new_y1 = b[9] - x1, b[10] - y1
                # new_x2, new_y2 = b[11] - x1, b[12] - y1
                # new_x3, new_y3 = b[7] - x1, b[8] - y1
                # new_x4, new_y4 = b[5] - x1, b[6] - y1
                new_x1, new_y1 = b[5] - x1, b[6] - y1
                new_x2, new_y2 = b[7] - x1, b[8] - y1
                new_x3, new_y3 = b[9] - x1, b[10] - y1
                new_x4, new_y4 = b[11] - x1, b[12] - y1
                print(new_x1, new_y1)
                print(new_x2, new_y2)
                print(new_x3, new_y3)
                print(new_x4, new_y4)
                        
                # 定义对应的点
                points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
                points2 = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])
                
                # 计算得到转换矩阵
                M = cv2.getPerspectiveTransform(points1, points2)
                
                # 实现透视变换转换
                processed = cv2.warpPerspective(img_raw, M, (CARD_WIDTH, CARD_HEIGHT))
                
                # 显示原图和处理后的图像
                # cv2.imshow("processed", processed)
                # save image
                # name = "test.jpg"
                name = os.path.basename(img_path)
                # cv2.imwrite(os.path.join(args.out_dir, name + f".{j}.processed.jpg"), processed)
                # cv2.imwrite(os.path.join(args.out_dir, name + f".{j}.show.jpg"), img_raw)
                # cv2.imwrite(os.path.join(args.out_dir, name + f".{j}.box.jpg"), img_box)
            name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(args.out_dir, f"{name}.show.minsize-640.jpg"), img_raw)
            # cv2.imshow('image', img_raw)
            # if cv2.waitKey(1000000) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_args())
