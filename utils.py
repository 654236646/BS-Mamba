import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F

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


def calculate_metric_percase(pred, gt):
    # 预处理，将预测和真实标签二值化
    pred = np.array(pred > 0, dtype=int)
    gt = np.array(gt > 0, dtype=int)

    # 计算 True Positives, False Positives, False Negatives, True Negatives
    true_positives = np.sum(pred * gt)
    false_positives = np.sum(pred * (1 - gt))
    false_negatives = np.sum((1 - pred) * gt)
    true_negatives = np.sum((1 - pred) * (1 - gt))


    # 计算指标
    dice = jaccard = precision = recall = accuracy = hd95 = mcc = 0

    if true_positives > 0:
        # 当有正确的正预测时
        dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        jaccard = true_positives / (true_positives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        # 计算 Matthews Correlation Coefficient (MCC)
        denominator = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) *
                              (true_negatives + false_positives) * (true_negatives + false_negatives))
        if denominator != 0:
            mcc = (true_positives * true_negatives - false_positives * false_negatives) / denominator
        else:
            mcc = 0  # 如果分母为0，手动设置 MCC 为 0

        # 限制 MCC 的范围，例如最大值为 1，最小值为 -1
        mcc = max(min(mcc, 1), -1)
    
        # 调用外部库计算 HD95
        hd95 = metric.binary.hd95(pred, gt)
    elif pred.sum() > 0:
        # 当有预测正类但无真实正类时
        precision = 1  # 预测为正但无真实正类
        recall = 0     # 召回率为0因为没有正确的正预测
        accuracy = true_negatives / (true_negatives + false_positives + false_negatives + true_positives)  # 只有负类
        mcc = 0  # 没有正确的正预测时，MCC为0
        hd95 = 0  # HD95定义为无穷大，因为理论上错误无限大
    elif gt.sum() > 0:
        # 当没有预测正类但有真实正类时
        precision = 0  # 没有预测为正的情况
        recall = 0     # 召回率为0因为没有正确的正预测
        accuracy = true_negatives / (true_negatives + false_positives + false_negatives + true_positives)  # 只有负类
        mcc = 0  # 没有正确的正预测时，MCC为0
    else:
        # 当没有预测正类也没有真实正类时
        accuracy = 1  # 完美的准确性，因为没有错误的预测
        mcc = 0  # 没有预测正类，MCC为0

    return dice, hd95, jaccard, precision, recall, accuracy, mcc


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
#
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
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
