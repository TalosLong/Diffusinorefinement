import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from sklearn.metrics import matthews_corrcoef
import numpy as np
from PIL import Image
import copy

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# MCC计算
def calculate_mcc(y_true, y_pred, smooth=1e-9):
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    mcc_score = (tp * tn - fp * fn) / np.sqrt((tp + fp + smooth) * (tp + fn + smooth) * (tn + fp + smooth) * (tn + fn + smooth))
    return mcc_score

# IoU计算
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_scores = np.sum(intersection) / np.sum(union)
    return iou_scores

#acc计算
def calculate_acc(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return acc

#sensitivity计算
def calculate_sen(y_true, y_pred): # 求敏感度 se=TP/(TP+FN)
    tp = np.sum(y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    SE = tp/(tp + fn)

    return SE  #返回batch中所有样本的SE和

#specificity计算
def calculate_spec(y_true, y_pred, threshold=0.5):#求特异性 sp=TN/(FP+TN)

    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)

    SP = tn/(fp + tn)

    return SP


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        metric_mcc = calculate_mcc(pred, gt)
        metric_iou = calculate_iou(pred, gt)
        acc = calculate_acc(pred, gt)
        sen = calculate_sen(pred, gt)
        spec = calculate_spec(pred, gt)

        return dice, hd95, metric_mcc, metric_iou, acc, sen, spec
    elif pred.sum() > 0 and gt.sum()==0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0, 0

#accuracy (Acc), sensitivity (Sen), specificity (Spec), Jaccard index (JI), and Dice coefficient (DC).

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _,x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)

        a1[a1 == 1] = 255
        a1[a1 == 2] = 0

        a2[a2 == 1] = 255
        a2[a2 == 2] = 0

        a3[a3 == 1] = 255
        a3[a3 == 2] = 0

        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path+'/'+case+'.png')


    return metric_list


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum()>0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum()==0:
#         return 1, 0
#     else:
#         return 0, 0


# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list