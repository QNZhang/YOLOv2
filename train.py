import time

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dataloaders.VOCdataloader import VOCdataset
from models.yolov2 import Yolov2
from misc.utils import Utils
from config import Config


class Trainer:
    """ based on github uvipen and tztztztztz """
    @staticmethod
    def train():

        print("Training process initialized...")
        print("dataset: ", Config.dataset_dir)

        dataset = VOCdataset(Config.dataset_dir, "2007", "train", Config.im_w)

        train_dataloader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=Config.num_workers,
                                      batch_size=Config.batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)

        print("lr:     ", Config.lr)
        print("batch:  ", Config.batch_size)
        print("epochs: ", Config.epochs)

        model = Yolov2()

        lr = Config.lr
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=Config.momentum, weight_decay=Config.weight_decay)

        starting_ep = 0

        if Config.continue_training:
            checkpoint = torch.load(Config.model_path)
            model.load_state_dict(checkpoint['model'])
            starting_ep = checkpoint['epoch'] + 1
            lr = checkpoint['lr']
            Trainer.adjust_learning_rate(optimizer, lr)

        model.cuda()
        model.train()

        counter = []
        loss_history = []
        iteration_number = 0

        best_loss = 10**15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        for epoch in range(starting_ep, Config.epochs):

            average_epoch_loss = 0
            count = 0
            start_time = time.time()

            if epoch in Config.decay_lrs:
                lr = Config.decay_lrs[epoch]
                Trainer.adjust_learning_rate(optimizer, lr)
                print('adjust learning rate to {}'.format(lr))

            for i, data in enumerate(train_dataloader, 0):

                im_data, boxes, classes, num_obj = data
                im_data, boxes, classes, num_obj = im_data.cuda(), boxes.cuda(), classes.cuda(), num_obj.cuda()

                im_data_variable = Variable(im_data).cuda()

                box_loss, iou_loss, class_loss = model(im_data_variable, boxes,
                                                       classes, num_obj, training=True)

                loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                average_epoch_loss += loss
                count += 1

            end_time = time.time() - start_time
            print("time: ", end_time)

            iteration_number += 1
            average_epoch_loss = average_epoch_loss / count

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(iteration_number)
            loss_history.append(loss.item())

            if average_epoch_loss < best_loss:
                best_epoch = epoch
                save_name = Config.best_model_path
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'lr': lr
                }, save_name)

                print("------------------------Best epoch: ", epoch)
                break_counter = 0

            save_name = Config.model_path
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'lr': lr
            }, save_name)

            if break_counter >= 20:
                print("Training break...")
                break

            break_counter += 1

        print("best: ", best_epoch)
        plt.plot(counter, loss_history)
        plt.show()

    @staticmethod
    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr







