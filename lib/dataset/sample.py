import json
import os

import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm


class SampleDataset(Dataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__()
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)

        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        # print(img_root)
        self.label_root = label_root
        self.mask_root = mask_root
        self.lane_root = lane_root
        self.db = self._get_db()

    def __len__(self, ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def _get_db(self):
        print('building database...')
        gt_db = []
        for img in tqdm(os.listdir(self.img_root)):
            image_path = os.path.join(self.img_root, img)
            lane_path = os.path.join(self.lane_root, os.path.basename(img)[:-10] + '000000.png')
            mask_path = os.path.join(self.mask_root, os.path.basename(img)[:-10] + '000000.png')
            gt = []
            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]
            gt_db += rec
        print('database build finish')
        return gt_db

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])  # cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
        else:
            seg_label = cv2.imread(data["mask"], 0)  # cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
        lane_label = cv2.imread(data["lane"], 0)

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True,
                                                             scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)

        det_label = data["label"]
        labels = []

        if len(det_label) > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        if self.is_train:
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            # print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S,
                        vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                lane_label = np.filpud(lane_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        if self.cfg.num_seg_class == 3:
            # _, seg0 = cv2.threshold(seg_label[:, :, 0], 128, 255, cv2.THRESH_BINARY)
            # _, seg1 = cv2.threshold(seg_label[:, :, 1], 1, 255, cv2.THRESH_BINARY)
            # _, seg2 = cv2.threshold(seg_label[:, :, 2], 1, 255, cv2.THRESH_BINARY)
            seg0 = np.where(seg_label[:, :, 0] == 0, 255, 0)
            seg1 = np.where(seg_label[:, :, 0] == 1, 255, 0)
            seg2 = np.where(seg_label[:, :, 0] == 2, 255, 0)
        else:
            _, seg1 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY_INV)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)  # 大于1的赋值255
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)
        #        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0

        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0], seg1[0], seg2[0]), 0).float()
        else:
            seg_label = torch.stack((seg2[0], seg1[0]), 0)

        lane_label = torch.stack((lane2[0], lane1[0]), 0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)

        target = [labels_out, seg_label, lane_label]
        img = self.transform(img)

        return img, target, data["image"], shapes


if __name__ == "__main__":
    sample = SampleDataset()
    print(sample[0])
