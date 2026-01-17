import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from loss.Losses import FocalLoss, TverskyLoss, BoundaryLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from test import inference

def trainer_synapse(args, model, snapshot_path, loss_alpha):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    focal_loss = FocalLoss()
    tversky_loss = TverskyLoss()
    boundary_loss = BoundaryLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # 打印参数量
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            # print(torch.min(label_batch), torch.max(label_batch))
            # print(torch.min(outputs), torch.max(outputs))
            # print(outputs.size())
            # print(label_batch.size())
            loss_ce = ce_loss(outputs, label_batch[:].long())#问题的源头
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            if args.loss_type == "DiceCE":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * ce_loss
            elif args.loss_type == "DiceFocal":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * focal_loss(outputs, label_batch[:].long())
            elif args.loss_type == "DiceBoundary":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * boundary_loss(outputs, label_batch[:].long())
            elif args.loss_type == "DiceTversky":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * tversky_loss(outputs, label_batch[:].long())
            # loss = loss_alpha * loss_ce + (1-loss_alpha) * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(),loss_ce.item()))# loss_ce.item()

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = max_epoch/6  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'CPUNet200midpepoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'CPUNet200endpepoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    logging.info("Evaluating model ...")
    model.eval()
    metric_list = []
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            image, label = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            dice, hd95, metric_mcc, metric_iou, acc, sen, spec= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      z_spacing=args.z_spacing)
            # 假设 test_single_volume 返回 {'Dice': x, 'IoU': y}
            metric_list.append([dice, hd95, metric_mcc, metric_iou, acc, sen, spec])
            

        metric_array = np.array(metric_list)
        avg_metrics = np.mean(metric_array, axis=0)
    logging.info(f"Average -> Dice {avg_metrics[0]:.4f}, HD95 {avg_metrics[1]:.4f}, IoU {avg_metrics[2]:.4f}, "
                 f"Precision {avg_metrics[3]:.4f}, Recall {avg_metrics[4]:.4f}, Spec {avg_metrics[5]:.4f}, Acc {avg_metrics[6]:.4f}")


    return avg_metrics
    

# def trainer_ISIC(args, model, snapshot_path):
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations
#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     print("The length of train set is: {}".format(len(db_train)))

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     # 打印参数量
#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("Total_params: {}".format(pytorch_total_params))
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())#问题的源头
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(),loss_ce.item()))# loss_ce.item()

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

#         save_interval = 8  # int(max_epoch/6)
#         if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#             save_mode_path = os.path.join(snapshot_path, 'ConvUnet200epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))

#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'ConvUnet200epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break

#     writer.close()
#     return "Training Finished!"
