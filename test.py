import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from dataloaders.VOCdataloader import VOCdataset
from config import Config
from models.yolov2 import Yolov2
from misc.utils import Utils

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
        mAP_path = "/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/"  # path to mAP repository
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

        # anchors width and height
        anchors = torch.cuda.FloatTensor(Config.anchors)

        net = Yolov2()
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.cuda()

        net.eval()

        cw = Config.im_w // Config.S
        ch = Config.im_h // Config.S

        tp = np.zeros(conf_threshs.size)
        tn = np.zeros(conf_threshs.size)
        fp = np.zeros(conf_threshs.size)
        fn = np.zeros(conf_threshs.size)

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                if (i % 5000 == 0):
                    print(i)  # progress

                img, targets, impath = data
                img = img.cuda()
                impath = impath[0].split("/")[len(impath[0].split("/")) - 1]

                cv_im = np.array(img[0].permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)

                predictions = net(img)

                # Decode model predictions
                pred_loc = torch.zeros_like(predictions[:, :, :, :, :4])
                pred_conf = torch.zeros_like(predictions[:, :, :, :, :1])

                pred_loc[:, :, :, :, :2] = torch.sigmoid(predictions[:, :, :, :, :2])
                pred_loc[:, :, :, :, 2:4] = torch.sigmoid(predictions[:, :, :, :, 2:4]) * 0.5
                pred_conf[:, :, :, :, 0] = torch.sigmoid(predictions[:, :, :, :, 4])
                pred_cls = predictions[:, :, :, :, 4:4 + dataset.num_classes]


                # decode pred boxes as x_center,y_center,w,h
                lin_x = torch.range(0, Config.S - 1).view(Config.S, -1).repeat(Config.S, Config.B).view(Config.S,Config.S,Config.B).cuda()
                lin_y = torch.range(0, Config.S - 1).view(Config.S, -1).repeat(Config.S, Config.B).view(Config.S,Config.S,Config.B).transpose(0,1).cuda()  # cell heights

                pred_boxes = torch.FloatTensor(Config.train_batch_size * Config.B * Config.S * Config.S, 4)

                pred_boxes[:, 0] = (pred_loc[:, :, :, :, 0].detach() + lin_x).contiguous().view(-1) * cw
                pred_boxes[:, 1] = (pred_loc[:, :, :, :, 1].detach() + lin_y).contiguous().view(-1) * ch
                pred_boxes[:, 2] = (pred_loc[:, :, :, :, 2].detach().exp() * anchors[:, 0]).view(-1) * cw
                pred_boxes[:, 3] = (pred_loc[:, :, :, :, 3].detach().exp() * anchors[:, 1]).view(-1) * ch
                pred_boxes = pred_boxes.cpu()

                # For predictions batch
                for b in range(pred_conf.data.size(0)):
                    # for thresholds (ROC)
                    for index in range(conf_threshs.size):
                        atp = 0
                        atn = 0
                        afp = 0
                        afn = 0

                        cur_pred_boxes = pred_boxes[b * (Config.B * Config.S * Config.S):(b + 1) * (Config.B * Config.S * Config.S)]
                        gt = torch.zeros(len(targets[b]), 4)

                        for i, anno in enumerate(targets[b]):
                            gt[i, 2] = (anno[2] - anno[0]) * 1  # Bug workaround: ( * 1)
                            gt[i, 3] = (anno[3] - anno[1]) * 1  # TypeError: can't assign a numpy.float32 to a Variable[CPUType]
                            gt[i, 0] = (anno[0] + gt[i, 2] / 2)
                            gt[i, 1] = (anno[1] + gt[i, 3] / 2)

                        ious = Utils.bbox_ious(gt, cur_pred_boxes)

                        if visualize:
                            cv_im = Tester.draw_detections(pred_loc, pred_conf, conf_threshs[index], cw, ch, pred_cls, dataset, targets, cv_im, b)

                        # mAP files
                        if save_mAP:
                            Tester.save_mAP_files(pred_loc, pred_conf, conf_threshs, cw, ch, pred_cls, dataset, targets, impath, b)

                        pred_cls = pred_cls.contiguous().view(-1, dataset.num_classes)
                        # if conf > conf_thresh
                        conf_mask = (pred_conf >= conf_threshs[index]).contiguous().view(-1)

                        # for each gt annotation
                        for i, annotation in enumerate(targets[b]):

                            # if ious > iou_thresh
                            iou_mask = ious[i] >= iou_thresh
                            # if conf and iou > threshs
                            mask = iou_mask.cuda() * conf_mask

                            if mask.sum(0) > 0:  # if any detections
                                for detect_cls in torch.argmax(pred_cls[mask][:], dim=1):
                                    if annotation[4] == detect_cls:
                                        atp += 1
                                    else:
                                        afp += 1  # fps caused by class errors
                            else:
                                afn += 1

                        fpmask = ious.sum(0) < iou_thresh
                        fpmask = fpmask.cuda() * conf_mask
                        afp += fpmask.sum(0)

                        # true rejection = all - (tp + fp + fn)
                        atn += max((Config.S * Config.S * Config.B) - (atp + afp + afn), 0)

                        tp[index] += atp
                        tn[index] += atn
                        fp[index] += afp
                        fn[index] += afn

                        if visualize:
                            print("tp: ", atp)
                            print("fp: ", afp.item())
                            print("fn: ", afn)
                            print("++ ")
                            cv2.imshow("test", cv_im)
                            cv2.waitKey(0)

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            tpr = np.append(tpr, 0.)
            fpr = np.append(fpr, 0.)

            #roc_auc = auc(fpr, tpr)
            #print("auc: ", roc_auc)

            print("tp: ", tp.tolist())
            print("tn: ", tn.tolist())
            print("fp: ", fp.tolist())
            print("fn: ", fn.tolist())
            print("tpr: ", tpr.tolist())
            print("fpr: ", fpr.tolist())

            print("FPsximage: ", (fpr * Config.S * Config.S * Config.B).tolist())

            fig = plt.figure()
            plt.plot(fpr, tpr, 'bo-')
            plt.plot([0, 1], [0, 1], 'k--')
            fig.suptitle('ROC curve')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.show()

    @staticmethod
    def draw_detections(pred_loc, pred_conf, conf_threshs, cw, ch, pred_cls, dataset, targets, cv_im, b):
        """ Draws the detections and gts for test visualization """
        # gts
        for target in targets[0]:
            x1 = target[0]
            y1 = target[1]
            x2 = target[2]
            y2 = target[3]
            cv2.rectangle(cv_im, (target[0], target[1]), (target[2], target[3]), color=(255, 0, 0), thickness=2)

        # detections
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
                        loc_x1 = max(0, loc_x1 - (loc_w / 2))
                        loc_y1 = max(0, loc_y1 - (loc_h / 2))
                        loc_x2 = min(loc_x1 + loc_w, Config.im_w)
                        loc_y2 = min(loc_y1 + loc_h, Config.im_h)

                        cv2.rectangle(cv_im, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                      color=(0, 0, 255), thickness=1)

                        cls_txt = dataset.classes[torch.argmax(pred_cls[b, h, w, a, :])]
                        cv_im = cv2.putText(cv_im, cls_txt, (loc_x1, loc_y1 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 255, 255), 2, cv2.LINE_AA)
        return cv_im

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

