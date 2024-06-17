import os
import random
import torch
import torch.utils.data as data
import cv2
import numpy as np
random.seed(0)


def rot_0(label):
    annotation = np.zeros((1, 13))
    # bbox
    annotation[0, 0] = label[2]  # x1
    annotation[0, 1] = label[3]  # y1
    annotation[0, 2] = label[4]  # x2
    annotation[0, 3] = label[5]  # y2

    # landmarks
    annotation[0, 4] = label[6]  # l0_x
    annotation[0, 5] = label[7]  # l0_y
    annotation[0, 6] = label[8]  # l1_x
    annotation[0, 7] = label[9]  # l1_y
    
    annotation[0, 8] = label[10]  # l3_x
    annotation[0, 9] = label[11]  # l3_y
    annotation[0, 10] = label[12]  # l4_x
    annotation[0, 11] = label[13]  # l4_y
    return annotation


def r90(x, y, w, h):
    return h - y, x


def r180(x, y, w, h):
    return w - x, h - y


def r270(x, y, w, h):
    return y, w - x


def rot_90(label, h0, w0):
    annotation = np.zeros((1, 13))
    # bbox
    annotation[0, 0], annotation[0, 1] = r90(label[2], label[3], w0, h0)
    annotation[0, 2], annotation[0, 3] = r90(label[4], label[5], w0, h0)
    # landmarks
    annotation[0, 4], annotation[0, 5] = r90(label[6], label[7], w0, h0)
    annotation[0, 6], annotation[0, 7] = r90(label[8], label[9], w0, h0)
    annotation[0, 8], annotation[0, 9] = r90(label[10], label[11], w0, h0)
    annotation[0, 10], annotation[0, 11] = r90(label[12], label[13], w0, h0)
    return annotation

def rot_180(label, h0, w0):
    annotation = np.zeros((1, 13))
    # bbox
    annotation[0, 0], annotation[0, 1] = r180(label[2], label[3], w0, h0)
    annotation[0, 2], annotation[0, 3] = r180(label[4], label[5], w0, h0)
    # landmarks
    annotation[0, 4], annotation[0, 5] = r180(label[6], label[7], w0, h0)
    annotation[0, 6], annotation[0, 7] = r180(label[8], label[9], w0, h0)
    annotation[0, 8], annotation[0, 9] = r180(label[10], label[11], w0, h0)
    annotation[0, 10], annotation[0, 11] = r180(label[12], label[13], w0, h0)
    return annotation

def rot_270(label, h0, w0):
    annotation = np.zeros((1, 13))
    # bbox
    annotation[0, 0], annotation[0, 1] = r270(label[2], label[3], w0, h0)
    annotation[0, 2], annotation[0, 3] = r270(label[4], label[5], w0, h0)
    # landmarks
    annotation[0, 4], annotation[0, 5] = r270(label[6], label[7], w0, h0)
    annotation[0, 6], annotation[0, 7] = r270(label[8], label[9], w0, h0)
    annotation[0, 8], annotation[0, 9] = r270(label[10], label[11], w0, h0)
    annotation[0, 10], annotation[0, 11] = r270(label[12], label[13], w0, h0)
    return annotation


class WiderCardSquareDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None, train_mode=True):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        self.train_mode = train_mode
        for line in lines:
            line = line.strip('\n').split(',')
            self.imgs_path.append(line[0])
            # lable = list(map(int, line[3:]))
            lable = list(map(int, line[1:]))
            self.words.append([lable]) 
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        h0, w0 = img.shape[:2]
        labels = self.words[index]
        # print(self.imgs_path[index], img.shape, labels)

        img_square = np.empty((w0 + h0, w0 + h0, 3), dtype=np.uint8)
        img_square[:, :] = (104, 117, 123)
        x0 = h0 // 2
        y0 = w0 // 2
        img_square[y0:y0+h0, x0:x0+w0] = img
        # cv2.imwrite(os.path.basename(self.imgs_path[index]) + '.square.jpg', img_square)
        # print(img_square.shape)
        '''
        if h0 > w0:
            img_square = np.zeros((h0, h0, 3), dtype=np.uint8)
            x0 = (h0 - w0) // 2
            img_square[:, x0:x0+w0, :] = img
            img = img_square
        elif w0 > h0:
            img_square = np.zeros((w0, w0, 3), dtype=np.uint8)
            y0 = (w0 - h0) // 2
            img_square[y0:y0+h0, :, :] = img
            img = img_square
        '''
        for j in range(len(labels)):
            '''
            if h0 > w0:
                x0 = (h0 - w0) // 2
                for i in range(2, 14, 2):
                    labels[j][i] += x0
            elif w0 > h0:
                y0 = (w0 - h0) // 2
                for i in range(3, 14, 2):
                    labels[j][i] += y0
            '''
            for i in range(2, 14, 2):
                labels[j][i] += x0
                labels[j][i + 1] += y0
        # print(self.imgs_path[index], img.shape, labels)
        '''
        if self.train_mode:
            rand_number = random.randint(0, 99) % 4
            if rand_number == 0:
                img = cv2.rotate(img, 0)
            elif rand_number == 1:
                img = cv2.rotate(img, 1)
            elif rand_number == 2:
                img = cv2.rotate(img, 2)
        '''
        annotations = np.zeros((0, 13))
        if len(labels) == 0:
            return annotations
        
        for idx, label in enumerate(labels):
            '''
            if self.train_mode:
                if rand_number == 0:
                    annotation = rot_90(label, h0, w0)
                elif rand_number == 1:
                    annotation = rot_180(label, h0, w0)
                elif rand_number == 2:
                    annotation = rot_270(label, h0, w0)
                else:
                    annotation = rot_0(label)
            else:
            '''
            annotation = rot_0(label)
            if (label[0] < 0):
                annotation[0, 12] = -1
            else:
                annotation[0, 12] = 1 
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations).astype(np.float64)
        # print(target)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        # print(img.shape)
        # print(target)
        return torch.from_numpy(img), target


if __name__ == '__main__':
    from data import detection_collate, preproc
    training_dataset = '/home/xin.zhang6/data_br/val.txt'
    img_dim = 640
    rgb_mean = (104, 117, 123)
    dataset_train = WiderCardSquareDetection(training_dataset, preproc(img_dim, rgb_mean), train_mode=True)
    print('dataset', len(dataset_train))
    batch_size = 2
    training_loader = data.DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0, collate_fn=detection_collate)
    for images, targets in training_loader:
        print(images.shape)
        print(targets.shape)
        break