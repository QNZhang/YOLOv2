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

        #folder_dataset = dset.ImageFolder(root=Config.training_dir)

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
            count = 0
            start_time = time.time()

            for i, data in enumerate(train_dataloader, 0):

                #print(i)
                #reset_time = time.time()

                img0, img1, targets, ima = data
                img0, img1 = img0.cuda(), img1.cuda()

                #print(type(ima[0]))

                optimizer.zero_grad()

                #print("Loading data: ", time.time() - reset_time)
                #reset_time = time.time()

                predictions = net(img0, img1)

                #print("predictions: ", time.time() - reset_time)
                #reset_time = time.time()

                loc_l, conf_l, noobj_conf_l, loss = criterion(predictions, targets)

                #print("Loss: ", time.time() - reset_time)
                #reset_time = time.time()

                loss.backward()
                optimizer.step()

                #print("backprop: ", time.time() - reset_time)

                average_epoch_loss += loss
                count += 1

                #print("p_mean: ", torch.mean(predictions))
                #print("p_std : ", torch.std(predictions))
                #print("mean xy:", torch.mean(predictions[:, :, :, :, :2]))
                #print("std xy: ", torch.std(predictions[:, :, :, :, :2]))
                #print("max xy: ", torch.max(predictions[:, :, :, :, :2]))
                #print("mean wh:", torch.mean(predictions[:, :, :, :, 2:4]))
                #print("std wh: ", torch.std(predictions[:, :, :, :, 2:4]))
                #print("max wh: ", torch.max(predictions[:, :, :, :, 2:4]))
                #print("conf mean: ", torch.mean(torch.sigmoid(predictions[:, :, :, :, 4])))
                #print("conf max : ", torch.max(torch.sigmoid(predictions[:, :, :, :, 4])))

                #print("=")
                #print("loss total: ", loss)
                #print("loss loc :  ", loc_l)
                #print("loss conf:  ", conf_l)
                #print("loss nconf: ", noobj_conf_l)
                #print("loss cls: ", cls_l)
                #print("====")
                #cv2.imshow("im", im)
                #cv2.imshow("iml", iml)
                #cv2.imshow("imc", imc)
                #cv2.imshow("imnc", imnc)
                #cv2.waitKey(1)

                #print(name0)
                #print(name1)

            end_time = time.time() - start_time
            print("time: ", end_time)

            iteration_number += 1
            average_epoch_loss = average_epoch_loss / count
            #print(count)

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