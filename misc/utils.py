
import cv2
import numpy as np
import matplotlib.pyplot as plt

from random import uniform
from torch.utils.data.dataloader import default_collate
import torch


class Utils:

    def custom_collate_fn(batch):
        """
            Collate data of different batch, it is because the boxes and gt_classes have changeable length.
            This function will pad the boxes and gt_classes with zero.

            Arguments:
            batch -- list of tuple (im, boxes, gt_classes)

            im_data -- tensor of shape (3, H, W)
            boxes -- tensor of shape (N, 4)
            gt_classes -- tensor of shape (N)
            num_obj -- tensor of shape (1)

            Returns:

            tuple
            1) tensor of shape (batch_size, 3, H, W)
            2) tensor of shape (batch_size, N, 4)
            3) tensor of shape (batch_size, N)
            4) tensor of shape (batch_size, 1)

            """

        # kind of hack, this will break down a list of tuple into
        # individual list
        bsize = len(batch)
        im_data, boxes, gt_classes, num_obj = zip(*batch)
        max_num_obj = max([x.item() for x in num_obj])
        padded_boxes = torch.zeros((bsize, max_num_obj, 4))
        padded_classes = torch.zeros((bsize, max_num_obj,))

        for i in range(bsize):
            padded_boxes[i, :num_obj[i], :] = boxes[i]
            padded_classes[i, :num_obj[i]] = gt_classes[i]

        for i in range(bsize):
            for box in range(padded_boxes.size()[1]):
                for coord in range(padded_boxes.size()[2]):
                    #print(padded_boxes[i][box][coord].item())
                    if padded_boxes[i][box][coord].item() >= 1:
                        print("error")

        return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)

    """ from: https://github.com/uvipen/Yolo-v2-pytorch """
    @staticmethod
    def bbox_ious(boxes1, boxes2):
        b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
        b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
        b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
        b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

        dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
        dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        unions = (areas1 + areas2.t()) - intersections

        return intersections / unions


""" Data augmentation utils from: https://github.com/uvipen/Yolo-v2-pytorch """


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Crop(object):

    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        for lb in label:
            xmin = min(xmin, lb[0])
            ymin = min(ymin, lb[1])
            xmax = max(xmax, lb[2])
            ymax = max(ymax, lb[2])
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        new_xmin = int(min(cropped_left * width, xmin))
        new_ymin = int(min(cropped_top * height, ymin))
        new_xmax = int(max(width - 1 - cropped_right * width, xmax))
        new_ymax = int(max(height - 1 - cropped_bottom * height, ymax))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        label = [[lb[0] - new_xmin, lb[1] - new_ymin, lb[2] - new_xmin, lb[3] - new_ymin] for lb in label]

        return image, label


class VerticalFlip(object):

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) >= self.prob:
            image = cv2.flip(image, 1)
            width = image.shape[1]
            label = [[width - lb[2], lb[1], width - lb[0], lb[3]] for lb in label]
        return image, label


class HSVAdjust(object):

    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):

        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        image, label = data
        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value
        image = image.astype(np.float32) / 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # modified
        image[:, :, 0] += adjust_hue
        image[:, :, 0] = clip_hue(image[:, :, 0])
        image[:, :, 1] = np.clip(adjust_saturation * image[:, :, 1], 0.0, 1.0)
        image[:, :, 2] = np.clip(adjust_value * image[:, :, 2], 0.0, 1.0)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # modified
        image = (image * 255).astype(np.float32)

        return image, label


class Resize(object):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size) / width
        height_ratio = float(self.image_size) / height
        new_label = []
        for lb in label:
            resized_xmin = lb[0] * width_ratio
            resized_ymin = lb[1] * height_ratio
            resized_xmax = lb[2] * width_ratio
            resized_ymax = lb[3] * height_ratio
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_label.append([int(resized_xmin), int(resized_ymin), int(resize_width), int(resize_height)])

        return image, new_label


class WeightLoader(object):
    """ https://github.com/tztztztztz/yolov2.pytorch """
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None

    def load_conv_bn(self, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)

        # make sure the loaded weight is right
        assert size == self.start