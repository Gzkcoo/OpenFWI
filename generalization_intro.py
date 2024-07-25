import numpy as np
import torch
import time
import torch.nn as nn
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data.dataloader import default_collate
from timeit import default_timer
import utils
import cv2
import json
import sys
import os
import transforms as T
import pytorch_ssim
import network
import datetime
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torchvision.transforms import Compose
from dataset import LDataset, FLDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

def l2_relative_loss(input, target):
    diff = input - target
    l2_norm_diff = torch.norm(diff, p=2, dim=(1, 2, 3))
    l2_norm_target = torch.norm(target, p=2, dim=(1, 2, 3))
    loss = l2_norm_diff / l2_norm_target
    return loss

def plot_velocity(output, target, source, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)

    # ax[0].set_xticks(range(0, 350, 50))
    # ax[0].set_xticklabels(range(0, 700, 100))
    # ax[0].set_yticks(range(0, 350, 50))
    # ax[0].set_yticklabels(range(0, 700, 100))
    # ax[0].set_ylabel('Depth (m)', fontsize=12)
    # ax[0].set_xlabel('Offset (m)', fontsize=12)
    #
    # ax[1].set_xticks(range(0, 70, 10))
    # ax[1].set_xticklabels(range(0, 700, 100))
    # ax[1].set_yticks(range(0, 70, 10))
    # ax[1].set_yticklabels(range(0, 700, 100))
    # ax[1].set_ylabel('Depth (m)', fontsize=12)
    # ax[1].set_xlabel('Offset (m)', fontsize=12)

    for axis in ax:
        # axis.set_xticks(range(0, 70, 10))
        # axis.set_xticklabels(range(0, 1050, 150))
        # axis.set_yticks(range(0, 70, 10))
        # axis.set_yticklabels(range(0, 1050, 150))
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))
        axis.set_ylabel('Depth (m)', fontsize=12)
        axis.set_xlabel(source, fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

def plot_four_velocity(ID_output, FD_output, IN_output, target, source, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))

    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    ax[0].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth', y=1.08, fontsize=22)
    ax[1].matshow(ID_output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Inversion-DeepONet', y=1.08, fontsize=22)
    ax[2].matshow(FD_output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title('Fourier-DeepONet', y=1.08, fontsize=22)
    im = ax[3].matshow(IN_output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[3].set_title('InversionNet', y=1.08, fontsize=22)

    # 定义五角星的坐标点
    # 0 172.5 345 517.5 690


    star_points = [(0, 0), (17, 0), (35, 0), (52, 0), (68, 0)]

    # 绘制五角星
    ax[0].scatter(*zip(*star_points), marker='*', s=100, c='red')


    # ax[0].set_xticks(range(0, 350, 50))
    # ax[0].set_xticklabels(range(0, 700, 100))
    # ax[0].set_yticks(range(0, 350, 50))
    # ax[0].set_yticklabels(range(0, 700, 100))
    # ax[0].set_ylabel('Depth (m)', fontsize=12)
    # ax[0].set_xlabel('Offset (m)', fontsize=12)
    #
    # ax[1].set_xticks(range(0, 70, 10))
    # ax[1].set_xticklabels(range(0, 700, 100))
    # ax[1].set_yticks(range(0, 70, 10))
    # ax[1].set_yticklabels(range(0, 700, 100))
    # ax[1].set_ylabel('Depth (m)', fontsize=12)
    # ax[1].set_xlabel('Offset (m)', fontsize=12)

    # for axis in ax:
    #     # axis.set_xticks(range(0, 70, 10))
    #     # axis.set_xticklabels(range(0, 1050, 150))
    #     # axis.set_yticks(range(0, 70, 10))
    #     # axis.set_yticklabels(range(0, 1050, 150))
    #     axis.set_xticks(range(0, 70, 10))
    #     axis.set_xticklabels(range(0, 700, 100))
    #     axis.set_yticks(range(0, 70, 10))
    #     axis.set_yticklabels(range(0, 700, 100))
    #     axis.set_ylabel('Depth (m)', fontsize=12)
    #     axis.set_xlabel('Length (m)', fontsize=12)

    ax[0].set_xticks(range(0, 70, 10))
    ax[0].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    ax[0].set_yticks(range(0, 70, 10))
    ax[0].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    ax[0].set_ylabel('Depth (km)', fontsize=18)
    ax[0].set_xlabel('Length (km)', fontsize=18)

    ax[1].set_xticks(range(0, 70, 10))
    ax[1].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    ax[1].set_yticks(range(0, 70, 10))
    ax[1].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    # ax[1].set_ylabel('Depth (m)', fontsize=12)
    ax[1].set_xlabel('Length (km)', fontsize=18)

    ax[2].set_xticks(range(0, 70, 10))
    ax[2].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    ax[2].set_yticks(range(0, 70, 10))
    ax[2].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    # ax[2].set_ylabel('Depth (m)', fontsize=12)
    ax[2].set_xlabel('Length (km)', fontsize=18)

    ax[3].set_xticks(range(0, 70, 10))
    ax[3].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    ax[3].set_yticks(range(0, 70, 10))
    ax[3].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=14)
    # ax[3].set_ylabel('Depth (m)', fontsize=12)
    ax[3].set_xlabel('Length (km)', fontsize=18)




    fig.supylabel("Random", fontsize=22, x=0.008)
    plt.tight_layout()
    font = {
            'size': 18
            }
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(label='Velocity(km/s)', fontdict=font)
    colorbar = np.around(cb.get_ticks() / 1000, decimals=1)
    cb.set_ticklabels(colorbar)
    plt.savefig(path)
    plt.close('all')

def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin

def tonumpy_MaxMindenormalize(vid, vmin, vmax, scale=2):
    vid = minmax_denormalize(vid.cpu().numpy(), vmin, vmax, scale)
    return vid

def plot_single_velocity(label, path):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vmax, vmin = np.max(label), np.min(label)
    im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    # im = ax.matshow(label, cmap="gist_rainbow", vmin=vmin, vmax=vmax)

    # nx = label.shape[0]
    # ax.set_aspect(aspect=1)
    # ax.set_xticks(range(0, nx, int(150//(1050/nx)))[:7])
    # ax.set_xticklabels(range(0, 1050, 150))
    # ax.set_yticks(range(0, nx, int(150//(1050/nx)))[:7])
    # ax.set_yticklabels(range(0, 1050, 150))
    # ax.set_title('Offset (m)', y=1.08)
    # ax.set_ylabel('Depth (m)', fontsize=18)

    # fig.colorbar(im, ax=ax, shrink=1.0, label='Velocity(km/s)')

    plt.xticks([])
    plt.yticks([])

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_histogarm(ID_RE, FD_RE, IN_RE):

    min_value = min(np.min(ID_RE), np.min(FD_RE), np.min(IN_RE))
    max_value = max(np.max(ID_RE), np.max(FD_RE), np.max(IN_RE))

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
    axs[0].hist(FD_RE, bins=35, range=(min_value, max_value), color='#F88455', alpha=0.7, edgecolor='white')
    axs[0].hist(ID_RE, bins=35, range=(min_value, max_value), color='#91BFFA', alpha=0.7, edgecolor='white')    # #91BFFA淡蓝  #F3ED99淡黄  #88CF73淡绿  #F8AF62
    axs[1].hist(IN_RE, bins=35, range=(min_value, max_value), color='#39B086', alpha=0.7, edgecolor='white')
    axs[1].hist(ID_RE, bins=35, range=(min_value, max_value), color='#91BFFA', alpha=0.7, edgecolor='white')

    # 设置标题和轴标签
    # axs[0].set_title('Inversion-DeepONet', fontsize=16)
    # axs[1].set_title('Fourier-DeepONet', fontsize=16)


    axs[1].set_xlabel('RE', fontsize=16)


    axs[0].set_ylabel('Frequency', fontsize=16)
    axs[1].set_ylabel('Frequency', fontsize=16)

    # axs[0].set_yticks(np.arange(0, axs[0].get_ylim()[1], 200))
    # axs[1].set_yticks(np.arange(0, axs[1].get_ylim()[1], 200))

    # 创建图例矩形对象
    legend_patches1 = [plt.Rectangle((0, 0), 1, 1, color='#F88455', alpha=0.7),
                      plt.Rectangle((0, 0), 1, 1, color='#91BFFA', alpha=0.7)]

    legend_patches2 = [plt.Rectangle((0, 0), 1, 1, color='#39B086', alpha=0.7),
                       plt.Rectangle((0, 0), 1, 1, color='#91BFFA', alpha=0.7)]


    axs[0].legend(legend_patches1, ['Fourier-DeepONet', 'Inversion-DeepONet'])
    axs[1].legend(legend_patches2, ['InversionNet', 'Inversion-DeepONet'])

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.savefig('./histogarm/histogarm.png')
    plt.close('all')

def evaluate(InversionNet, criterion, dataloader, device, k, ctx,
                vis_path, vis_batch, vis_sample, missing, std, with_source):

    InversionNet.eval()

    label_list, ID_label_pred_list, FD_label_pred_list, IN_label_pred_list = [], [], [], [] # store denormalized predcition & gt in numpy
    label_tensor, ID_label_pred_tensor, FD_label_pred_tensor, IN_label_pred_tensor = [], [], [], [] # store normalized prediction & gt in tensor
    source_np_list = []


    with torch.no_grad():
        batch_idx = 0
        for data, source, label in dataloader:
            data = data.to(device, non_blocking=True)
            source = source.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)

            IN_pred = InversionNet(data)

            # if with_source == 1:
            #     pred = model(data, source)
            # else:
            #     pred = model(data)

            source_np = tonumpy_MaxMindenormalize(source, 5, 25)
            source_np_list.append(source_np)

            IN_label_pred_np = T.tonumpy_denormalize(IN_pred, ctx['label_min'], ctx['label_max'], exp=False)
            IN_label_pred_list.append(IN_label_pred_np)
            IN_label_pred_tensor.append(IN_pred)

            # Visualization
            if vis_path and batch_idx < vis_batch * 8:
                for i in range(vis_sample):
                    # plot_single_velocity(label_np[i, 0],f'{vis_path}/V_{batch_idx}_{i}_GT.png')

                    plot_single_velocity(IN_label_pred_np[i, 0], f'{vis_path}/V_{batch_idx}_{i}_14.svg')
                    # plot_four_velocity(ID_label_pred_np[i, 0], FD_label_pred_np[i, 0], IN_label_pred_np[i, 0], label_np[i, 0],
                    #                    source_np[i], f'{vis_path}/V_{batch_idx}_{i}_{np.around(source_np[i], decimals=1)}.png')  # , vmin=ctx['label_min'], vmax=ctx['label_max'])
            batch_idx += 8
    #l1 = nn.L1Loss()
    #l2 = nn.MSELoss()
    ## label, ID_label_pred, FD_label_pred, IN_label_pred = np.concatenate(label_list), np.concatenate(ID_label_pred_list),\
    ##                                                      np.concatenate(FD_label_pred_list), np.concatenate(IN_label_pred_list)
    #label_t, ID_pred_t, FD_pred_t, IN_pred_t = torch.cat(label_tensor), torch.cat(ID_label_pred_tensor), \
    #                                           torch.cat(FD_label_pred_tensor), torch.cat(IN_label_pred_tensor)

    #source_list = np.concatenate(source_np_list)
    #print("Inversion_DeepONet---------------------------------------------------")
    #print(f'MAE: {l1(label_t, ID_pred_t)}')
    #print(f'MSE: {l2(label_t, ID_pred_t)}')
    #ssim_loss = pytorch_ssim.SSIM(window_size=11)
    #print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, ID_pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)
    #ID_RE = l2_relative_loss(label_t, ID_pred_t)
    #print(f'RE: {torch.mean(ID_RE)}')
#
    #print("Fourier_DeepONet---------------------------------------------------")
    #print(f'MAE: {l1(label_t, FD_pred_t)}')
    #print(f'MSE: {l2(label_t, FD_pred_t)}')
    #ssim_loss = pytorch_ssim.SSIM(window_size=11)
    #print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, FD_pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)
    #FD_RE = l2_relative_loss(label_t, FD_pred_t)
    #print(f'RE: {torch.mean(FD_RE)}')
#
    #print("InversionNet---------------------------------------------------")
    #print(f'MAE: {l1(label_t, IN_pred_t)}')
    #print(f'MSE: {l2(label_t, IN_pred_t)}')
    #ssim_loss = pytorch_ssim.SSIM(window_size=11)
    #print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, IN_pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)
    #IN_RE = l2_relative_loss(label_t, IN_pred_t)
    #print(f'RE: {torch.mean(IN_RE)}')
#
    #plot_histogarm(ID_RE.cpu().numpy(), FD_RE.cpu().numpy(), IN_RE.cpu().numpy())
    # max_index = np.argsort(FD_RE.cpu().numpy() / ID_RE.cpu().numpy() + IN_RE.cpu().numpy() / ID_RE.cpu().numpy())[-20:]

    # for i in max_index:
    #     plot_four_velocity(ID_pred_t.cpu().numpy()[i, 0], FD_pred_t.cpu().numpy()[i, 0], IN_pred_t.cpu().numpy()[i, 0], label_t.cpu().numpy()[i, 0],
    #                        np.around(source_list[i], decimals=0),
    #                        f'{vis_path}/VV_{i}_{np.around(source_list[i], decimals=0)}.png')  # , vmin=ctx['label_min'], vmax=ctx['label_max'])


def test(args):

    utils.mkdir(args.output_path)
    utils.mkdir('histogarm')
    torch.backends.cudnn.benchmark = True

    print(args)

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    ctx['file_size'] = 500
    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=1))
    ])
    # 震源位置归一化
    # transform_source = Compose([
    #     T.MinMaxNormalize(0, 700)
    # ])
    # 震源频率归一化
    transform_source = Compose([
        T.MinMaxNormalize(5, 25)
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])


    print('Loading validation data')
    # Normalize data and label to [-1, 1]
    dataset_valid = LDataset(
        args.val_anno,
        preload=True,
        sample_ratio=1,
        file_size=ctx['file_size'],
        transform_data=transform_data,
        transform_source=transform_source,
        transform_label=transform_label
    )

    # dataset_all = LDataset(
    #     args.val_anno,
    #     preload=True,
    #     sample_ratio=1,
    #     file_size=ctx['file_size'],
    #     transform_data=transform_data,
    #     transform_source=transform_source,
    #     transform_label=transform_label,
    #     du=args.data_url
    # )
    #
    # dataset_train, dataset_valid = torch.utils.data.random_split(dataset=dataset_all,
    #                                                              lengths=[int(len(dataset_all) / 9 * 8),
    #                                                                       int(len(dataset_all) / 9)],
    #                                                              generator=torch.Generator().manual_seed(
    #                                                                  0))  # Normalize data and label to [-1, 1]

    valid_sampler = SequentialSampler(dataset_valid)


    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    # model = network.model_dict[args.model](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)

    # Inversion_DeepONet = network.model_dict['InversionDeepOnet'](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)
    # Fourier_DeepONet = network.model_dict['FourierDeeponet'](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)
    InversionNet = network.model_dict['InversionNet'](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)

    # checkpoint = torch.load('./trained_model/FWI_F/CurveVel_A/Inversion-DeepONet/checkpoint.pth', map_location='cpu')
    # Inversion_DeepONet.load_state_dict(network.replace_legacy(checkpoint['model']))
    # checkpoint = torch.load('./trained_model/FWI_F/CurveVel_A/Fourier-DeepONet/checkpoint.pth', map_location='cpu')
    # Fourier_DeepONet.load_state_dict(network.replace_legacy(checkpoint['model']))
    #checkpoint = torch.load('./trained_model/FWI_F/FlatVel_B/InversionNet/checkpoint.pth', map_location='cpu')
    checkpoint = torch.load('./pretrained/CurveVel_A/checkpoint.pth', map_location='cpu')
    InversionNet.load_state_dict(network.replace_legacy(checkpoint['model']))

    # FWIDeepONet = network.model_dict['FWIDeeponet'](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)
    # checkpoint =  torch.load('./trained_model/FWI_F/CurveVel_A/FWIDeepONet/checkpoint.pth', map_location='cpu')
    # FWIDeepONet.load_state_dict(network.replace_legacy(checkpoint['model']))
    #
    # DDNet = network.model_dict['DDNetDeepOnet'](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)
    # checkpoint =  torch.load('./trained_model/FWI_F/CurveVel_A/DDNetDeepONet/checkpoint.pth', map_location='cpu')
    # DDNet.load_state_dict(network.replace_legacy(checkpoint['model']))

    # print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))



    # Define loss function
    l1loss = nn.L1Loss()  # MAE
    l2loss = nn.MSELoss()  # MSE

    # l3loss = network.BCEFocalLoss()  # 交叉熵

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)

        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    # if args.resume:
    #     print(args.resume)
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(network.replace_legacy(checkpoint['model']))
    #     print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))

    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.train_out, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None


    print('Start testing')
    evaluate(InversionNet, criterion, dataloader_valid, args.device, args.k, ctx,
             vis_path, args.vis_batch, args.vis_sample, args.missing, args.std, args.with_source)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvevel-a', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val_anno', default='generalization.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('--with_source', default=1, type=int, help='data with source or not')
    parser.add_argument('-m', '--model', default='InversionDeepOnet', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    # Test related
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    # 服务器
    # parser.add_argument('-r', '--resume', default='model-out/checkpoint.pth', help='resume from checkpoint')
    # 本地
    parser.add_argument('-r', '--resume', default='checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_false")
    parser.add_argument('-vsu','--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb','--vis-batch', help='number of batch to be visualized', default=10, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=6, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[5, 256, 256, 256, 512], help='trunk_net layer sizes')

    # nuaa
    parser.add_argument('--data_url', type=str, default='')  # data_url
    parser.add_argument('--train_model_out', type=str, default='')  # 模型输出路径
    parser.add_argument('--train_out', type=str, default='')  # 文件输出路径
    parser.add_argument('--train_visualized_log', type=str, default='')  # 可视化路径日志
    parser.add_argument('--gpu_num_per_node', type=int, default=1)  # GPU节点个数
    parser.add_argument('--model_load_dir', type=str, default='')  # 预训练模型路径


    args = parser.parse_args()

    # 服务器
    # args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    # args.val_anno = os.path.join(args.anno_path, args.val_anno)
    # args.resume = os.path.join(args.model_load_dir, args.resume)

    # 本地
    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)




    return args





if __name__ == '__main__':
    args = parse_args()
    start_time = default_timer()
    test(args)
    end_time = default_timer()
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(end_time-start_time)))))

