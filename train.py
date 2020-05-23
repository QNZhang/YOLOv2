import time
import cv2

import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch import optim

from dataloaders.VOCdataloader import VOCdataset
from models.DarkNet19 import DarkNet19
from loss.YOLOloss import LossMN
from misc.utils import Utils
from config import Config


class Trainer:

    @staticmethod
    def train():

        torch.cuda.manual_seed(123)

        print("Training process initialized...")
        print("dataset: ", Config.training_dir)

        dataset = VOCdataset(Config.training_dir, "2012", "train", Config.im_w)

        train_dataloader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=8,
                                      batch_size=Config.train_batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)

        print("lr:     ", Config.lrate)
        print("batch:  ", Config.train_batch_size)
        print("epochs: ", Config.train_number_epochs)

        net = DarkNet19(dataset.num_classes, Config.anchors)

        optimizer = optim.SGD(net.parameters(), lr=Config.lrate, momentum=0.9, weight_decay=0.0005)

        starting_ep = 0

        if Config.continue_training:
            print("Continue training:")
            checkpoint = torch.load(Config.model_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_ep = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print("epoch: ", starting_ep, ", loss: ", loss)

        net.cuda()
        net.train()

        criterion = LossMN(dataset.num_classes, Config.anchors)

        counter = []
        loss_history = []
        iteration_number = 0

        best_loss = 10**15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        for epoch in range(starting_ep, Config.train_number_epochs):

            average_epoch_loss = 0
            average_loc_l = 0
            average_conf_l = 0
            average_cls_l = 0
            start_time = time.time()

            for i, data in enumerate(train_dataloader, 0):

                #print(i)
                #reset_time = time.time()

                img, targets = data
                img = img.cuda()

                optimizer.zero_grad()

                predictions = net(img)

                loss, loc_l, conf_l, cls_l, cv_im = criterion(predictions, targets, img)

                loss.backward()
                optimizer.step()

                average_epoch_loss += loss
                average_loc_l += loc_l
                average_conf_l += conf_l
                average_cls_l += cls_l

                cv2.imshow("im", cv_im)
                #cv2.imshow("iml", iml)
                #cv2.imshow("imc", imc)
                #cv2.imshow("imnc", imnc)
                cv2.waitKey(1)

            end_time = time.time() - start_time
            print("time: ", end_time)

            iteration_number += 1
            average_epoch_loss = average_epoch_loss / i
            print("ll : ", average_loc_l / i)
            print("cl : ", average_conf_l / i)
            print("nll: ", average_cls_l / i)

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(iteration_number)
            loss_history.append(loss.item())

            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, Config.best_model_path)

                print("------------------------Best epoch: ", epoch)
                break_counter = 0
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, Config.model_path)

            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        print("best: ", best_epoch)
        Utils.show_plot(counter, loss_history)