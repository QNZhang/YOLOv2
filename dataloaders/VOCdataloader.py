""" from: https://github.com/uvipen/Yolo-v2-pytorch """

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from config import Config
from torch.utils.data import Dataset
from misc.utils import Compose, HSVAdjust, VerticalFlip, Crop, Resize


class VOCdataset(Dataset):

    # TODO: review the loading parameters (init) for VOC 2007-2012
    def __init__(self, root_path="data/VOCdevkit", year="2007", mode="train", image_size=448, is_training = True):

        self.mode = mode
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

        size_elem = annot.find("size")
        im_w = int(size_elem.find("width").text)
        im_h = int(size_elem.find("height").text)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())

            w_ratio = Config.im_w / im_w
            h_ratio = Config.im_h / im_h
            xmin = int(xmin * w_ratio)
            xmax = int(xmax * w_ratio)
            ymin = int(ymin * h_ratio)
            ymax = int(ymax * h_ratio)

            objects.append([xmin, ymin, xmax, ymax, label])

        image = cv2.resize(image, (Config.im_w, Config.im_h))

        if self.mode == "val":
            return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32), image_path
        else:
            return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
