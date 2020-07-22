import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.VOCdataloader import VOCdataset
from config import Config
from models.yolov2 import Yolov2
from misc.utils import Utils
from misc.yolo_eval import yolo_eval

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from sklearn.metrics import auc


class Tester:

    @staticmethod
    def test():

        print("testing...")

        Config.train_batch_size = 1
        conf_threshs = np.linspace(0, 1, 20)
        conf_threshs = np.array([0.1])
        iou_thresh = 0.3

        visualize = False  # Show the image with gts and predictions
        save_mAP = True  # save the files for mAP computation, requites to fix the path
        mAP_path = "/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/"  # output path for mAP files
        # Do not use save_mAP with an array of conf_threshs. Suitable for a single thresh value

        model_path = Config.best_model_path

        print("model: ", model_path)
        print("conf: ", conf_threshs.tolist())
        print("iou thresh:  ", iou_thresh)

        dataset = VOCdataset(Config.training_dir, "2012", "val", Config.im_w)

        dataloader = DataLoader(dataset,
                                      shuffle=False,
                                      num_workers=0,
                                      batch_size=Config.train_batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)

        model = Yolov2()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.cuda()
        model.eval()

        dataset_size = len(dataloader.image_index)
        all_boxes = [[[] for _ in range(dataset_size)] for _ in range(dataloader.num_classes)]

        img_id = -1

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):
                img_id += 1

                if (i % 5000 == 0):
                    print(i)  # progress

                img, boxes, classes, num_obj = data
                img, boxes, classes, num_obj = img.cuda(), boxes.cuda(), classes.cuda(), num_obj.cuda()

                im_data_variable = Variable(img).cuda()

                yolo_outputs = model(im_data_variable)

                output = [item[i].data for item in yolo_outputs]
                detections = yolo_eval(output, conf_threshold=Config.conf_thresh,
                                       nms_threshold=Config.nms_thresh)

                if len(detections) > 0:
                    for cls in range(dataloader.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                # mAP files
                #if save_mAP:
                #    Tester.save_mAP_files(pred_loc, pred_conf, conf_threshs, cw, ch, pred_cls, dataset, targets, impath, b)

    @staticmethod
    def save_mAP_files(pred_loc, pred_conf, conf_threshs, cw, ch, pred_cls, dataset, targets, impath, b):
        """ Saves the detections and gts for later mAP computation using:
         https://github.com/rafaelpadilla/Object-Detection-Metrics"""

        detection_str = ""
        gt_str = ""

        for anno in targets[0]:
            left = anno[0]
            top = anno[1]
            right = anno[2]
            bottom = anno[3]
            gt_str += dataset.classes[int(anno[4])] + " " \
                      + str(left.item()) + " " \
                      + str(top.item()) + " " \
                      + str(right.item()) + " " \
                      + str(bottom.item()) + "\n"
        f = open("/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/groundtruths/" + impath.split(".")[
            0] + ".txt", "a+")
        f.seek(0)
        if not (gt_str in f.readlines()):
            f.write(gt_str)
        f.close()

        for w in range(pred_loc.data.size(1)):  # grid w (S)
            for h in range(pred_loc.data.size(2)):  # grid h (S)
                for a in range(pred_loc.data.size(3)):  # Anchors (B)

                    # get cell
                    cell_x = w * cw
                    cell_y = h * ch

                    if pred_conf[b][h][w][a][0].cpu().numpy() >= conf_threshs:
                        # decode and format (x1,y1,x2,y2):
                        loc_x1 = (pred_loc[b][h][w][a][0] * cw) + cell_x
                        loc_y1 = (pred_loc[b][h][w][a][1] * ch) + cell_y
                        loc_w = (pred_loc[b][h][w][a][2]).exp() * Config.anchors[a][0] * cw
                        loc_h = (pred_loc[b][h][w][a][3]).exp() * Config.anchors[a][1] * ch
                        left = max(0, (loc_x1 - (loc_w / 2)).item())
                        top = max(0, (loc_y1 - (loc_h / 2)).item())
                        right = min((loc_x1 + loc_w).item(), Config.im_w)
                        bottom = min((loc_y1 + loc_h).item(), Config.im_h)

                        detection_str += dataset.classes[torch.argmax(pred_cls[b, h, w, a, :])] + " " \
                                         + str(pred_conf[b, h, w, a, 0].item()) + " " \
                                         + str(left) + " " \
                                         + str(top) + " " \
                                         + str(right) + " " \
                                         + str(bottom) + "\n"
        f = open("/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/detections/" + impath.split(".")[
            0] + ".txt", "a+")
        f.seek(0)
        if not (detection_str in f.readlines()):
            f.write(detection_str)
        f.close()