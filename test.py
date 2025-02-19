import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from Architecture.BS_Mamba import BS_Mamba as BS_seg
from Architecture.BS_Mamba import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=130, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    # Initialize lists to store metrics for averaging later
    dice_scores = []
    hd95_scores = []
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    mcc_scores = []  # Add a list for MCC

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metrics = test_single_volume(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

        # Assume metrics returns a list of tuples for each class: (dice, hd95, jaccard, precision, recall, accuracy, mcc)
        # Aggregate metrics
        for m in metrics:
            dice_scores.append(m[0])
            hd95_scores.append(m[1])
            jaccard_scores.append(m[2])
            precision_scores.append(m[3])
            recall_scores.append(m[4])
            accuracy_scores.append(m[5])
            mcc_scores.append(m[6])  # Add MCC to the list

        # Log metrics for this batch
        logging.info(
            f'idx {i_batch} case {case_name} - Dice: {m[0]:.3f}, HD95: {m[1]:.3f}, Jaccard: {m[2]:.3f}, Precision: {m[3]:.3f}, Recall: {m[4]:.3f}, Accuracy: {m[5]:.3f}, MCC: {m[6]:.3f}')

    # Calculate and log the mean metrics after processing all images
    mean_dice = np.mean(dice_scores)
    mean_hd95 = np.mean(hd95_scores)
    mean_jaccard = np.mean(jaccard_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_mcc = np.mean(mcc_scores)  # Calculate the mean MCC

    logging.info(
        f'Testing performance in best val model -  Dice: {mean_dice:.5f},  HD95: {mean_hd95:.5f},  Jaccard: {mean_jaccard:.5f},  Precision: {mean_precision:.5f},  Recall: {mean_recall:.5f},  Accuracy: {mean_accuracy:.5f},  MCC: {mean_mcc:.5f}')

    return "Testing Finished!"


if __name__ == "__main__":

    if args.is_savenii:
        print('Saving results as NIfTI files.')
    else:
        print('Not saving results.')

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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'BS-Mamba_Ablation_None' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'BUS-A2')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = BS_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))

    snapshot = '/hy-tmp/BSMamba/model/BS-Mamba-0127Synapse224/BUSI_R50-ViT-B_16_skip3_vitpatch16_30k_epo130_bs12_lr0.001_224_s3234/epoch_129.pth'
    print(snapshot)
    # net.load_state_dict(torch.load(snapshot, map_location='cuda:0'))
    # 加载预训练模型权重的路径
    pretrained_path = snapshot
    # 加载预训练模型的状态字典
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # 过滤掉不匹配的权重并加载
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留模型中存在的键
    model_dict.update(pretrained_dict)  # 更新模型的状态字典
    net.load_state_dict(model_dict, strict=False)  # 加载权重，strict=False允许不匹配的键
    # print(net)
    # summary = summary(net, input_size=( 3, 224, 224))
    # print(summary)

    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/0127' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.is_savenii = True

    if args.is_savenii:
        args.test_save_dir = './BSMamba_HEBefore'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


