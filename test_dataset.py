import torch
import torch.utils
import torch.utils.data as data
from data import WiderLPDetection, detection_collate, preproc
from data import WiderCardDetection
from data import WiderCardSquareDetection


if __name__ == '__main__':
    training_dataset = '/home/xin.zhang6/data_br/val.txt'
    # import sys
    # sys.path.append('..')
    # from data_augment import preproc
    img_dim = 640
    rgb_mean = (104, 117, 123),
    dataset = WiderCardSquareDetection(training_dataset, preproc(img_dim, rgb_mean), train_mode=False)
    loader = data.DataLoader(dataset, 2, shuffle=True, num_workers=0, collate_fn=detection_collate)
    for imgs, targets in loader:
        print(imgs)
        print(targets)
        break