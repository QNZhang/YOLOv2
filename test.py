import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from sklearn.metrics import auc

from dataloaders.VOCdataloader import VOCdataset
from config import Config
from models.yolov2 import Yolov2
from misc.utils import Utils
from misc.yolo_eval import yolo_eval
from misc import eval


class Tester:

    @staticmethod
    def test():

        print("testing...")

        Config.batch_size = 1
        conf_threshs = np.linspace(0, 1, 20)
        conf_threshs = np.array([0.5])
        iou_thresh = 0.5

        output_dir = "output"

        visualize = False  # Show the image with gts and predictions
        save_mAP = True  # save the files for mAP computation, requites to fix the path
        mAP_path = "/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/"  # path to mAP repository
        # Do not use save_mAP with an array of conf_threshs. Suitable for a single thresh value

        model_path = Config.best_model_path

        print("model: ", model_path)
        print("conf: ", conf_threshs.tolist())
        print("iou thresh:  ", iou_thresh)

        dataset = VOCdataset(Config.training_dir, "2012", "val", Config.im_w, is_training=False)

        dataloader = DataLoader(dataset,
                                      shuffle=False,
                                      num_workers=0,
                                      batch_size=Config.batch_size,
                                      drop_last=True)

        model = Yolov2()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.cuda()
        model.eval()

        all_boxes = [[[] for _ in range(len(dataloader))] for _ in range(dataset.num_classes)]
        img_id = -1

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                img_id += 1

                if (i % 5000 == 0):
                    print(i)  # progress

                im_data, im_infos = data
                im_data_variable = Variable(im_data).cuda()

                yolo_outputs = model(im_data_variable)

                im_info = {'width': im_infos[0][0], 'height': im_infos[0][1]}
                output = [item[0].data for item in yolo_outputs]

                detections = yolo_eval(output, im_info, conf_threshold=Config.conf_thresh,
                                       nms_threshold=Config.nms_thresh)

                if len(detections) > 0:
                    for cls in range(dataset.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

        eval.evaluate_detections(all_boxes, output_dir=output_dir)


