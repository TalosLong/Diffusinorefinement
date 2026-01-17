import argparse
import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.ConvUnet import ConvUnet as ConvUnet
# from networks.ConvTrans_spatial import ConvUnet_spatial as ConvUSnet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.ConvUnet_s2mlp import ConvUnet_mlp as con_s2
# from networks.ConvUnet_s2mlp_SFA import ConvUnet_s2mlp_SFA as con_s2_sfa
from networks.TransUnet_s2mlp import TransUnet_mlp as transunet_s2
from loss.Losses import FocalLoss, TverskyLoss, BoundaryLoss
# from trainer import trainer_synapse
import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/Synapse/train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=350, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.002,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--loss_alpha', type=float, 
                    default=0.5, help='weight for Dice loss, CE loss weight will be 1-alpha')
parser.add_argument('--loss_type', type=str, 
                    default='DiceCE', help='loss type ')
parser.add_argument('--scheduler', type=str, 
                    default='poly', help='scheduler type ')
parser.add_argument('--opti', type=str, 
                    default='SGD', help='optimizer type ')
args = parser.parse_args()

def search_best_threshold(outputs, labels, thresholds=np.arange(0.3, 0.91, 0.05)):
    """
    outputs: torch.Tensor, shape [B, 2, H, W] -> softmax概率
    labels: torch.Tensor, shape [B, H, W]
    thresholds: list of threshold values to test
    """
    probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # 前景类概率
    best_dice = 0.0
    best_th = 0.5
    for th in thresholds:
        preds = (probs > th).float()
        dice = dice_coefficient(preds, labels)
        if dice > best_dice:
            best_dice = dice
            best_th = th
    return best_th, best_dice

def dice_coefficient(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * inter + eps) / (union + eps)
    return dice.item()

def validate(model, val_loader, classes):
    from utils import calculate_metric_percase
    model.eval()
    total_metric = np.zeros(7)  # 假设 calculate_metric_percase 返回4个指标
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch['image'].cuda(), batch['label'].cuda()  # 根据你的dataset修改ke
            outputs = model(images)
            outputs_argmax = torch.argmax(torch.softmax(outputs, dim=1), dim=1) # [H,W]
            outputs_np = outputs_argmax.cpu().detach().numpy()
            # 转 numpy 计算指标
            # outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(images.shape[0]):  # batch 内每个样本
                metric = []
                for c in range(1, classes):
                    metric.append(calculate_metric_percase(outputs_np[i] == c, labels_np[i] == c))
                total_metric += np.mean(metric, axis=0)
                count += 1

    return total_metric / count

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    返回一个 LambdaLR 学习率调度器，实现 Warmup + Cosine Annealing。
    num_cycles=0.5 表示半个周期（cosine从1降到0）。
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup阶段：线性增加学习率
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def trainer_synapse(args, model, snapshot_path, loss_alpha):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from torch.utils.data import random_split
    train_dataset = Synapse_dataset(base_dir=args.root_path + "/train", list_dir=args.list_dir, split="train",transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    total_size = len(train_dataset)
    print("The length of train set is: {}".format(len(train_dataset)))
    val_size = total_size // 10  # 10%
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(42)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # val_dataset = Synapse_dataset(base_dir=args.root_path + "/val", list_dir=args.list_dir, split="val",transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    # valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                           worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    epsilon = 0.1
    focal_loss = FocalLoss()
    tversky_loss = TverskyLoss()
    boundary_loss = BoundaryLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    if args.opti == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opti == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr = base_lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    warmup_iters = int(0.1 * max_iterations)  # warmup 10%
    if args.scheduler == "poly":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: (1 - it / max_iterations) ** 0.9)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iters, num_training_steps=max_iterations)
    # Warmup + Cosine Annealing
    # warmup_iters = int(0.1 * max_iterations)  # warmup 10%
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iters, num_training_steps=max_iterations)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # 打印参数量
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    best_val_metric = 0.0
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
            # loss = loss_alpha * loss_ce + (1-loss_alpha) * loss_dice
            if args.loss_type == "DiceCE":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * loss_ce
            elif args.loss_type == "DiceFocal":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * focal_loss(outputs, label_batch[:].long())
            elif args.loss_type == "DiceBoundary":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * boundary_loss(outputs, label_batch[:].long())
            elif args.loss_type == "DiceTversky":
                loss = loss_alpha * loss_dice + (1 - loss_alpha) * tversky_loss(outputs, label_batch[:].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Warmup + Cosine Annealing
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
                # 更新学习率（每个iteration调用）
            scheduler.step()
            # current_lr = scheduler.get_last_lr()[0]
        # 
            #CPUnet
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', current_lr, iter_num)
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

        #     # **验证阶段**
        val_metric = validate(model, valloader, classes=args.num_classes)
        logging.info(f"Validation metrics: {val_metric}")

        model.train()

        # **保存最佳模型**
        mean_dice = val_metric[0]  # 假设第一个是Dice
        if mean_dice > best_val_metric :
            print("I am in!")
            best_val_metric = mean_dice
            best_epoch = epoch_num + 1
            torch.save(model.state_dict(), os.path.join(snapshot_path,   'PIM Ablation' + str(args.loss_type) +'_'+ str(args.opti) +'_'+ str(args.loss_alpha) + 'test_TransUNet_s2_tooth_300_best.pth'))
            logging.info(f"New best model saved at epoch {epoch_num+1} with Dice {mean_dice:.4f}")

        # save_interval = max_epoch/6  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path + 'finetuning-' + str(args.loss_alpha) + 'CPUNet_val_CVC_20midepoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path,'PIM Ablation-' +  str(args.loss_type) +'_'+ str(args.opti) + '_'+ str(args.loss_alpha) + 'test_TransUNet_s2_tooth_end_' + str(epoch_num+1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return f"New best model saved at epoch {epoch_num+1} with Dice {mean_dice:.4f}"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/home/yyc/Public/lhy/TransUNet/data/Synapse-tooth/',
            'list_dir': '/home/yyc/Public/lhy/TransUNet/lists/lists_Synapse-tooth',
            'num_classes': 2,
        },
        'ISIC': {
            'root_path': '/home/lhy/code/TransUNet/data/ISIC/train',
            'list_dir': '/home/lhy/code/TransUNet/lists/ISIC_list/',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    # heads=8, dim_head=64, dropout=0., num_patches=1024
    config_vit.dim = 3
    config_vit.heads = 8
    config_vit.dim_head = 64
    config_vit.dropout = 0
    config_vit.num_patches = 1024
    config_vit.mlp_dim = 1024
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net = ConvUnet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()# bl + convformer
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()# baseline
    # net = ConvUSnet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()# convUNet+spatial
    # net = con_s2(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()# convUNet + S2mlp*2
    net = transunet_s2(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()# TransUNet + S2mlp*2
    # net = con_s2_sfa(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()#convUnet+s2mlp+sfm
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, loss_alpha=args.loss_alpha)