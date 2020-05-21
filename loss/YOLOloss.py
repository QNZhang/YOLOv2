import math
import torch
import torch.nn as nn
from config import Config


class LossMN(torch.nn.Module):
    """
    Inspired from:
    - https://github.com/nhviet1009/Yolo-v2-pytorch/
    - https://github.com/kuangliu/pytorch-yolov2
    """

    def __init__(self, num_classes, anchors):
        super(LossMN, self).__init__()

        self.num_classes = num_classes
        self.anchors = anchors

    def forward(self, model_output, target):

        loc_preds, conf_preds, cls_preds,loc_targets, conf_targets, cls_targets,\
        loc_mask, conf_mask, noobj_conf_mask, cls_mask = self.decode(model_output, target)

        # Compute losses
        mse = nn.MSELoss(size_average=False)
        ce = nn.CrossEntropyLoss(size_average=False)

        loss_loc = Config.alpha_coord * mse(loc_preds * loc_mask, loc_targets * loc_mask) / Config.train_batch_size

        loss_conf = (mse(conf_preds * conf_mask, conf_targets * conf_mask) +
                        (Config.alpha_noobj * mse(conf_preds * noobj_conf_mask, conf_targets * noobj_conf_mask))) \
                        / Config.train_batch_size

        loss_cls = Config.alpha_cls * 2 * ce(cls_preds, cls_targets) / Config.train_batch_size

        loss_tot = self.loss_loc + self.loss_conf + self.loss_cls

        return loss_tot, loss_loc, loss_conf, loss_cls

    def decode(self, pred_vec, target_vec):
        # Cell size
        cw = Config.im_w // Config.S
        ch = Config.im_h // Config.S

        # anchors width and height
        anchors = torch.cuda.FloatTensor(Config.anchors)

        # Create tensors for locs, confs ans masks
        pred_loc = torch.zeros_like(pred_vec[:, :, :, :, :4])
        pred_conf = torch.zeros_like(pred_vec[:, :, :, :, :1])
        pred_cls = torch.zeros_like(pred_vec[:, :, :, :, :self.num_classes])
        target_loc = torch.zeros_like(pred_loc)
        target_conf = torch.zeros_like(pred_conf)
        target_cls = torch.zeros_like(pred_cls)

        loc_mask = torch.zeros_like(pred_loc, requires_grad=False)
        conf_mask = torch.zeros_like(pred_conf, requires_grad=False)
        noobj_conf_mask = torch.ones_like(pred_conf, requires_grad=False)
        cls_mask = torch.zeros_like(pred_cls, requires_grad=False)

        # Decode model predictions
        pred_loc[:, :, :, :, :2] = torch.sigmoid(pred_vec[:, :, :, :, :2])
        pred_loc[:, :, :, :, 2:4] = torch.sigmoid(pred_vec[:, :, :, :, 2:4]) * 0.5
        pred_conf[:, :, :, :, 0] = torch.sigmoid(pred_vec[:, :, :, :, 4])
        pred_cls[:, :, :, :, 0:self.num_classes] = pred_vec[:, :, :, 4:4+self.num_classes].contiguous()\
            .view(Config.train_batch_size, Config.S * Config.S,self.anchors, self.num_classes).contiguous()\
            .view(-1, self.num_classes)


        # create all prediction boxes in pred_vec
        lin_x = torch.range(0, Config.S - 1).view(Config.S, -1).repeat(Config.S, Config.B).view(Config.S, Config.S,
                                                                                                Config.B).cuda()
        lin_y = torch.range(0, Config.S - 1).view(Config.S, -1).repeat(Config.S, Config.B).view(Config.S, Config.S,
                                                                                                Config.B).transpose(0,
                                                                                                                    1).cuda()  # cell heights
        pred_boxes = torch.FloatTensor(Config.train_batch_size * Config.B * Config.S * Config.S, 4)

        pred_boxes[:, 0] = (pred_loc[:, :, :, :, 0].detach() + lin_x).contiguous().view(-1) * cw
        pred_boxes[:, 1] = (pred_loc[:, :, :, :, 1].detach() + lin_y).contiguous().view(-1) * ch
        pred_boxes[:, 2] = (pred_loc[:, :, :, :, 2].detach().exp() * anchors[:, 0]).view(-1) * cw
        pred_boxes[:, 3] = (pred_loc[:, :, :, :, 3].detach().exp() * anchors[:, 1]).view(-1) * ch
        pred_boxes = pred_boxes.cpu()

        for b in range(pred_vec.data.size(0)):

            # calc ious for boxes in prediction for current batch
            cur_pred_boxes = pred_boxes[
                             b * (Config.B * Config.S * Config.S):(b + 1) * (Config.B * Config.S * Config.S)]
            gt = torch.zeros(len(target_vec[b][target_vec[b][:, 4] == 1]), 4)

            aux_cont = 0
            for i, anno in enumerate(target_vec[b]):
                if anno[4] == 1:
                    gt[aux_cont, 0] = (anno[0] + anno[2] / 2)
                    gt[aux_cont, 1] = (anno[1] + anno[3] / 2)
                    gt[aux_cont, 2] = anno[2] - anno[0]
                    gt[aux_cont, 3] = anno[3] * anno[1]
                    aux_cont += 1

            iou_gt_pred = LossMN.bbox_ious(gt, cur_pred_boxes)

            # Set the conf mask to 1 for prediction boxes with iou > thresh
            mask = (iou_gt_pred > Config.iou_thresh).sum(0) >= 1
            mask = mask.view_as(conf_mask[b])

            conf_mask[b][mask] = 1
            target_conf[b][mask] = 1
            noobj_conf_mask[b][mask] = 0

            aux_cont = 0
            for i, annotation in enumerate(target_vec[b]):

                if annotation[4] == 1:
                    # print(annotation)
                    iou_gt = iou_gt_pred[aux_cont].view_as(conf_mask[b])  # .permute(1, 0, 2, 3)

                    max_index = torch.argmax(iou_gt)

                    ra = max_index % Config.B
                    rw = (max_index / Config.B) % Config.S
                    rh = (max_index / (Config.B * Config.S)) % Config.S

                    target_x = annotation[0] + (annotation[2] / 2)
                    target_y = annotation[1] + (annotation[3] / 2)
                    target_w = annotation[2] / cw
                    target_h = annotation[3] / ch

                    # Target loc for the highest IOU anchor
                    target_loc[b][rh][rw][ra][0] = float(target_x - (rw * cw)) / cw
                    target_loc[b][rh][rw][ra][1] = float(target_y - (rh * ch)) / ch

                    aux_x = (target_w / anchors[ra][0])
                    aux_y = (target_h / anchors[ra][1])
                    target_loc[b][rh][rw][ra][2] = math.log(aux_x)  # w
                    target_loc[b][rh][rw][ra][3] = math.log(aux_y)  # h

                    # Set the loc mask
                    loc_mask[b, rh, rw, ra, 0] = 1
                    loc_mask[b, rh, rw, ra, 1] = 1
                    loc_mask[b, rh, rw, ra, 2] = 1
                    loc_mask[b, rh, rw, ra, 3] = 1

                    aux_cont += 1

        pred_loc = pred_loc.cuda()
        pred_conf = pred_conf.cuda()
        pred_cls = pred_cls.cuda()
        target_loc = target_loc.cuda()
        target_conf = target_conf.cuda()
        target_cls = target_cls.cuda()
        loc_mask = loc_mask.cuda()
        conf_mask = conf_mask.cuda()
        # cls_mask = cls_mask.cuda()
        noobj_conf_mask = noobj_conf_mask.cuda()

        # return pred_loc, pred_conf, pred_cls, target_loc, target_conf,\
        #       target_cls, loc_mask, conf_mask, noobj_conf_mask, cls_mask
        return pred_loc, pred_conf, target_loc, target_conf, loc_mask, conf_mask, noobj_conf_mask

    def get_responsible_anchor(self, target_vec, pred_loc, cw, ch, batch):

        # rw = (x1 + w) / cw
        rw = (target_vec[0] + (target_vec[2] / 2)) // cw
        rh = (target_vec[1] + (target_vec[3] / 2)) // ch
        rw, rh = int(rw.item()), int(rh.item())

        targ_x1 = target_vec[0]
        targ_y1 = target_vec[1]
        targ_x2 = target_vec[0] + target_vec[2]
        targ_y2 = target_vec[1] + target_vec[3]

        center_x = (rw * cw) + (cw / 2)
        center_y = (rh * ch) + (ch / 2)

        max_iou = 0
        best_a = 0
        cont = 0
        for a in Config.anchors:
            """
            a_x1 = center_x - ((cw * a[0]) / 2)
            a_y1 = center_y - ((ch * a[1]) / 2)
            a_x2 = center_x + ((cw * a[0]) / 2)
            a_y2 = center_y + ((ch * a[1]) / 2)
            """
            # pred
            if rw > 14 or rh > 14:
                print("fck target")

            a_x1 = ((pred_loc[batch][rw][rh][cont][0] * cw).detach() + (rw * cw)).item()
            a_y1 = ((pred_loc[batch][rw][rh][cont][1] * ch).detach() + (rh * ch)).item()
            a_w = (pred_loc[batch][rw][rh][cont][2].detach().exp() * a[0] * cw).item()
            a_h = (pred_loc[batch][rw][rh][cont][3].detach().exp() * a[1] * ch).item()
            # center:
            a_x1 = max(0, a_x1 - (a_w / 2))
            a_y1 = max(0, a_y1 - (a_h / 2))
            a_x2 = min(a_x1 + a_w, Config.im_w)
            a_y2 = min(a_y1 + a_h, Config.im_h)

            iou = self.calc_iou(a_x1, a_y1, a_x2, a_y2, targ_x1, targ_y1, targ_x2, targ_y2)

            if iou > max_iou:
                max_iou = iou
                best_a = cont
            cont += 1

        return rw, rh, best_a

    @staticmethod
    def calc_iou(x1, y1, x1b, y1b, x2, y2, x2b, y2b):

        dx = min(x1b, x2b) - max(x1, x2)
        dy = min(y1b, y2b) - max(y1, y2)

        if (dx >= 0) and (dy >= 0):
            intersect_area = dx * dy

            area1 = (x1 - x1b) * (y1 - y1b)
            area2 = (x2 - x2b) * (y2 - y2b)

            return intersect_area / float(area1 + area2 - intersect_area)

        else:
            return 0

    # TODO: use from utils
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

    @staticmethod
    def test_masks(loc_preds, conf_preds, loc_targets, conf_targets, loc_mask, conf_mask, noobj_conf_mask, im):

        import cv2
        import numpy as np

        im_out = cv2.imread(im[0])  # np.zeros((Config.im_w, Config.im_h, 3), np.uint8)
        im_out = cv2.resize(im_out, (Config.im_w, Config.im_h))

        cw = Config.im_w // Config.S
        ch = Config.im_h // Config.S

        """  
        lmask_im = np.zeros((Config.im_w, Config.im_h, 3), np.uint8)
        cmask_im = np.zeros((Config.im_w, Config.im_h, 3), np.uint8)
        ncmask_im = np.zeros((Config.im_w, Config.im_h, 3), np.uint8)
        for b in range(1):  # loc_mask.data.size(0)):
            for w in range(loc_mask.data.size(1)):  # grid w (S)
                for h in range(loc_mask.data.size(2)):  # grid h (S)
                    cx = cw * w
                    cy = ch * h
                    cont = 0
                    for a in Config.anchors:
                        if loc_mask[b][w][h][cont][0] == 1:
                            cv2.rectangle(lmask_im, (cx, cy), (cx + cw, cy + ch),
                                          color=(255, 255, 255), thickness=-1)
                        if conf_mask[b][w][h][cont][0] == 1:
                            cv2.rectangle(cmask_im, (cx, cy), (cx + cw, cy + ch),
                                          color=(255, 255, 255), thickness=-1)
                        if noobj_conf_mask[b][w][h][cont][0] == 0:
                            cv2.rectangle(ncmask_im, (cx, cy), (cx + cw, cy + ch),
                                          color=(255, 255, 0), thickness=-1)
                            loc_x1 = (loc_preds[b][w][h][cont][0] * cw) + (w * cw)
                            loc_y1 = (loc_preds[b][w][h][cont][1] * ch) + (h * ch)
                            loc_w = (loc_preds[b][w][h][cont][2]).exp() * a[0] * cw
                            loc_h = loc_preds[b][w][h][cont][3].exp() * a[1] * ch
                            # center:
                            loc_x1 = max(0, loc_x1 - (loc_w / 2))
                            loc_y1 = max(0, loc_y1 - (loc_h / 2))
                            loc_x2 = min(loc_x1 + loc_w, Config.im_w)
                            loc_y2 = min(loc_y1 + loc_h, Config.im_h)
                            cv2.rectangle(im_out, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                          color=(255, 0, 255), thickness=2)
                        cont += 1
        #"""
        for b in range(1):  # loc_mask.data.size(0)):
            for w in range(loc_mask.data.size(1)):  # grid w (S)
                for h in range(loc_mask.data.size(2)):  # grid h (S)

                    cx = cw * w
                    cy = ch * h

                    test = -1
                    cont = 0
                    for a in Config.anchors:
                        if loc_mask[b][w][h][cont][0] == 1:
                            test = cont
                        if conf_preds[b][w][h][cont][
                            0] > Config.iou_thresh:  # and noobj_conf_mask[b][w][h][cont][0] > 0:
                            # Draws all false predictions
                            loc_x1 = (loc_preds[b][w][h][cont][0] * cw) + (w * cw)
                            loc_y1 = (loc_preds[b][w][h][cont][1] * ch) + (h * ch)
                            loc_w = (loc_preds[b][w][h][cont][2]).exp() * a[0] * cw
                            loc_h = loc_preds[b][w][h][cont][3].exp() * a[1] * ch
                            # center:
                            loc_x1 = max(0, loc_x1 - (loc_w / 2))
                            loc_y1 = max(0, loc_y1 - (loc_h / 2))
                            loc_x2 = min(loc_x1 + loc_w, Config.im_w)
                            loc_y2 = min(loc_y1 + loc_h, Config.im_h)

                            if noobj_conf_mask[b][w][h][cont][0] > 0:
                                cv2.rectangle(im_out, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                              color=(0, 0, 255), thickness=2)
                            else:
                                cv2.rectangle(im_out, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                              color=(255, 0, 255), thickness=2)
                        cont += 1

                    if test != -1:

                        # Draws cell
                        cv2.rectangle(im_out, (cx, cy), (cx + cw, cy + ch),
                                      color=(150, 150, 150), thickness=3)

                        cont = 0
                        for a in Config.anchors:
                            # Draws the anchors of the responsible cell
                            loc_x1 = (loc_preds[b][w][h][cont][0] * cw) + (w * cw)
                            loc_y1 = (loc_preds[b][w][h][cont][1] * ch) + (h * ch)
                            loc_w = (loc_preds[b][w][h][cont][2]).exp() * a[0] * cw
                            loc_h = loc_preds[b][w][h][cont][3].exp() * a[1] * ch
                            # center:
                            loc_x1 = max(0, loc_x1 - (loc_w / 2))
                            loc_y1 = max(0, loc_y1 - (loc_h / 2))
                            loc_x2 = min(loc_x1 + loc_w, Config.im_w)
                            loc_y2 = min(loc_y1 + loc_h, Config.im_h)

                            # print("a0:", (loc_x1, loc_y1), "a1:", (loc_x2, loc_y2))
                            cv2.rectangle(im_out, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                          color=(0, 255, 0), thickness=1)
                            cont += 1

                        # Draws the responsible anchor
                        loc_x1 = (loc_preds[b][w][h][test][0] * cw) + (w * cw)
                        loc_y1 = (loc_preds[b][w][h][test][1] * ch) + (h * ch)
                        loc_w = (loc_preds[b][w][h][test][2]).exp() * Config.anchors[test][0] * cw
                        loc_h = loc_preds[b][w][h][test][3].exp() * Config.anchors[test][1] * ch
                        # center:
                        loc_x1 = max(0, loc_x1 - (loc_w / 2))
                        loc_y1 = max(0, loc_y1 - (loc_h / 2))
                        loc_x2 = min(loc_x1 + loc_w, Config.im_w)
                        loc_y2 = min(loc_y1 + loc_h, Config.im_h)

                        # print("w:  ", w)
                        # print("h:  ", h)
                        # print("cw: ", cw)
                        # print((loc_x1, loc_y1), (loc_x2, loc_y2))
                        cv2.rectangle(im_out, (loc_x1, loc_y1), (loc_x2, loc_y2),
                                      color=(0, 255, 0), thickness=3)

                        targ_x1 = (loc_targets[b][w][h][test][0] * cw) + (w * cw)
                        targ_y1 = (loc_targets[b][w][h][test][1] * ch) + (h * ch)
                        targ_w = (loc_targets[b][w][h][test][2]).exp() * Config.anchors[test][0] * cw
                        targ_h = loc_targets[b][w][h][test][3].exp() * Config.anchors[test][1] * ch
                        # center:
                        targ_x1 = max(0, targ_x1 - (targ_w / 2))
                        targ_y1 = max(0, targ_y1 - (targ_h / 2))
                        targ_x2 = min(targ_x1 + targ_w, Config.im_w)
                        targ_y2 = min(targ_y1 + targ_h, Config.im_h)
                        # print((targ_x1, targ_y1), (targ_x2, targ_y2))
                        cv2.rectangle(im_out, (targ_x1, targ_y1), (targ_x2, targ_y2),
                                      color=(255, 0, 0), thickness=3)
                    else:
                        # cell
                        cv2.rectangle(im_out, (cx, cy), (cx + cw, cy + ch),
                                      color=(50, 50, 50), thickness=1)

        # blank_image = cv2.resize(blank_image, (800, 800))
        # cv2.imshow("a", blank_image)
        # cv2.waitKey()
        # print("test on loss test_masks")
        return im_out
        # return im_out, lmask_im, cmask_im, ncmask_im