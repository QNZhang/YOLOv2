""" from: https://github.com/uvipen/Yolo-v2-pytorch """

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from misc.utils import Compose, HSVAdjust, VerticalFlip, Crop, Resize


class VOCdataset(Dataset):

    # TODO: review the loading parameters (init) for VOC 2007-2012
    def __init__(self, root_path="data/VOCdevkit", year="2007", mode="train", image_size=448, is_training = True):

        if (mode in ["train", "val", "trainval", "test"] and year == "2007") or (
                mode in ["train", "val", "trainval"] and year == "2012"):
            self.data_path = os.path.join(root_path, "VOC{}".format(year))
        id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
        self.ids = [id.strip() for id in open(id_list_path)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        annot = ET.parse(image_xml_path)

        boxes = []
        classes = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(label)

        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])

        image, boxes = transformations((image, boxes))

        w, h, _ = image.shape
        boxes = np.array(boxes, dtype=np.float32)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

        im_data = torch.from_numpy(np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))) / 255
        boxes = torch.from_numpy(boxes)
        classes = torch.from_numpy(np.array(classes, dtype=np.int32))
        num_objs = torch.Tensor([boxes.size(0)]).long()

        for box in range(boxes.size()[0]):
            for coord in range(boxes.size()[1]):

                if boxes[box][coord].item() >= 1:
                    print("error")

        #return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
        return im_data, boxes, classes, num_objs
