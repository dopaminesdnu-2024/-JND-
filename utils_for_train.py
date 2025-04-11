#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: HCN
@time: 2023/10/14 0014 18:28
@work unit:SDNU
@email: dopaminesdnu@163.com
@describe:  collection of methods for main.py
"""
import logging
import os.path
import random
import time
from functools import partial

import torch
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import ini  # 一些公共包
from model.DISTS_pt import DISTS

# dataset
from my_dataset.dataset_kadid10k import KadidDataset
# from my_dataset.dataset2_tid2013 import Tid2013Dataset  # TODO 逐元素最小搜索没有使用
from my_dataset.dataset_tid2013 import Tid2013Dataset
from my_dataset.dataset_csiq import CsiqDataset
from my_dataset.dataset_live import LiveDataset
# from my_dataset.dataset_tid2013_ntimes import Tid2013Dataset_ntimes
#  TODO 新的plan3 增强
# from my_dataset.dataset_tid2013_ntimes3 import Tid2013Dataset


def get_transforms(model='vgg16'):
    """对于图像transforms的设置


    Parameters
    ----------
        model: net
        is_train: bool
            默认True, 即训练模式的transforms

    Returns
    -------
        transforms:
            pytorch Sequential
    """
    transforms = 0
    if model == 'vgg16' or model == 'resnet18' or model == 'resnet34' or model == 'resnet50':
        transforms = ini.Compose([
            ini.Resize(size=(256, 256), antialias=True),
            ini.Normalize(0.5, 0.5)]
        )
    elif model == 'dists':
        transforms = ini.Compose([
            ini.Resize(size=256)])
        pass
    return transforms


def get_datasets(is_train=True, transforms=None, mode: str = 'base', n_t=1, device=None, *dataset_names):
    """通过数据集名称获得对应的Dataset类
        已包含数据集 kadid10k tid2013 100% 50% 25% 10%
    Parameters
    ----------
        mode: str
        是否增强, base 不增强, aug 增强\
        注意 is_train 是 True and mode = aug 才增强
        is_train: bool
            该数据集是train 还是 test, 默认 True即 train
            只是打印训练集信息,可以忽略作用,正常输入即可
        transforms: torchvision.transforms
            该数据集使用的transforms
        dataset_names: tuple
            数据集的名称

    Returns
    -------
        字典
            该字典 key是数据集的名称, value是dataset类
    """

    # dataset_names : ([]) 对于vgg等特别处理
    if isinstance(dataset_names[0], list):
        dataset_names = dataset_names[0]
    aug = False
    dev = device
    if mode == 'aug' and is_train:  # 数据增强并且是训练集
        aug = True
    datasets_dict = {}
    for elem_idx, elem_name in enumerate(dataset_names):
        # 不划分数据集, IQA训练 # TODO, aug参数未起作用
        if elem_name == 'kadid10k':
            kadid10k_dataset = KadidDataset(transform=transforms, aug=aug)
            datasets_dict[elem_name] = kadid10k_dataset  # add to dict
        elif elem_name == 'tid2013':
            tid2013_dataset = Tid2013Dataset(transform=transforms, aug=aug)
            datasets_dict[elem_name] = tid2013_dataset  # add to dict
        elif elem_name == 'csiq':
            csiq_dataset = CsiqDataset(transform=transforms)
            datasets_dict[elem_name] = csiq_dataset  # add to dict
        elif elem_name == 'live':
            live_dataset = LiveDataset(transform=transforms)
            datasets_dict[elem_name] = live_dataset  # add to dict

        # 数据集随机划分的情况 通常用于backbone(vgg16, resnet...)训练
        # kadid10k_08  mean 100% kadid10k, 八二划分训练测试集
        # kadid10k_50_08 mean 50% kadid10k, 八二划分训练测试集
        # kadid10k_25_08 mean 25% kadid10k, 八二划分训练测试集
        elif elem_name == 'kadid10k_08':
            kadid10k_08dataset = KadidDataset(transform=transforms, size=1.0, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = kadid10k_08dataset  # add to dict
        elif elem_name == 'kadid10k_50_08':
            kadid10k_50_08dataset = KadidDataset(transform=transforms, size=0.5, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = kadid10k_50_08dataset  # add to dict
        elif elem_name == 'kadid10k_25_08':
            kadid10k_25_08dataset = KadidDataset(transform=transforms, size=0.25, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = kadid10k_25_08dataset  # add to dict
        elif elem_name == 'kadid10k_10_08':
            kadid10k_10_08dataset = KadidDataset(transform=transforms, size=0.1, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = kadid10k_10_08dataset  # add to dict

        elif elem_name == 'kadid10k_02':
            kadid10k_02dataset = KadidDataset(transform=transforms, size=1.0, split=0.2)
            datasets_dict[elem_name] = kadid10k_02dataset  # add to dict
        elif elem_name == 'kadid10k_50_02':
            kadid10k_50_02dataset = KadidDataset(transform=transforms, size=0.5, split=0.2)
            datasets_dict[elem_name] = kadid10k_50_02dataset  # add to dict
        elif elem_name == 'kadid10k_25_02':
            kadid10k_25_02dataset = KadidDataset(transform=transforms, size=0.25, split=0.2)
            datasets_dict[elem_name] = kadid10k_25_02dataset  # add to dict
        elif elem_name == 'kadid10k_10_02':
            kadid10k_10_02dataset = KadidDataset(transform=transforms, size=0.1, split=0.2)
            datasets_dict[elem_name] = kadid10k_10_02dataset  # add to dict

        elif elem_name == 'tid2013_08':
            tid2013_08dataset = Tid2013Dataset(transform=transforms, size=1.0, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = tid2013_08dataset  # add to dict
        elif elem_name == 'tid2013_50_08':
            tid2013_50_08dataset = Tid2013Dataset(transform=transforms, size=0.5, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = tid2013_50_08dataset  # add to dict
        elif elem_name == 'tid2013_25_08':
            tid2013_25_08dataset = Tid2013Dataset(transform=transforms, size=0.25, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = tid2013_25_08dataset  # add to dict
        elif elem_name == 'tid2013_10_08':
            tid2013_10_08dataset = Tid2013Dataset(transform=transforms, size=0.1, split=0.8, aug=aug, device=dev)
            datasets_dict[elem_name] = tid2013_10_08dataset  # add to dict

        elif elem_name == 'tid2013_02':
            tid2013_02dataset = Tid2013Dataset(transform=transforms, size=1.0, n_t=n_t, split=0.2)
            datasets_dict[elem_name] = tid2013_02dataset  # add to dict
        elif elem_name == 'tid2013_50_02':
            tid2013_50_02dataset = Tid2013Dataset(transform=transforms, size=0.5, split=0.2)
            datasets_dict[elem_name] = tid2013_50_02dataset  # add to dict
        elif elem_name == 'tid2013_25_02':
            tid2013_25_02dataset = Tid2013Dataset(transform=transforms, size=0.25, split=0.2)
            datasets_dict[elem_name] = tid2013_25_02dataset  # add to dict
        elif elem_name == 'tid2013_10_02':
            tid2013_10_02dataset = Tid2013Dataset(transform=transforms, size=0.1, split=0.2)
            datasets_dict[elem_name] = tid2013_10_02dataset  # add to dict

        # TID2013train数据集扩充的dataset
        ###
        # 训练集命名
        #     tid2013_08_ntimes  其中n 代表train 扩大n倍
        # 测试集命名
        #     全部tid2013_600n  因为测试集不变，无论训练集怎么扩大在meta_info中最后600条数据永远是测试集

        # elif elem_name == 'tid2013_08_2times':
        #     tid2013_08_2times = Tid2013Dataset_ntimes(transforms, aug=aug, times=2, device=dev)
        #     datasets_dict[elem_name] = tid2013_08_2times
        # elif elem_name == 'tid2013_08_5times':
        #     tid2013_08_5times = Tid2013Dataset_ntimes(transforms, aug=aug, times=5, device=dev)
        #     datasets_dict[elem_name] = tid2013_08_5times
        # elif elem_name == 'tid2013_08_10times':
        #     tid2013_08_10times = Tid2013Dataset_ntimes(transforms, aug=aug, times=10, device=dev)
        #     datasets_dict[elem_name] = tid2013_08_10times
        # elif elem_name == 'tid2013_08_15times':
        #     tid2013_08_15times = Tid2013Dataset_ntimes(transforms, aug=aug, times=15, device=dev)
        #     datasets_dict[elem_name] = tid2013_08_15times
        #
        # elif elem_name == 'tid2013_08_20times':
        #     tid2013_08_20times = Tid2013Dataset_ntimes(transforms, aug=aug, times=20, device=dev)
        #     datasets_dict[elem_name] = tid2013_08_20times
        #
        # elif elem_name == 'tid2013_600n':
        #     tid2013_600n = Tid2013Dataset_ntimes(transforms)
        #     datasets_dict[elem_name] = tid2013_600n
        else:
            raise ValueError('给定的数据集名称不存在!!!重新输入')

    if is_train:
        print(f"训练集: {datasets_dict.keys()}")
    else:
        print(f"测试集: {datasets_dict.keys()}")

    return datasets_dict


def get_dataloaders(configs, is_train=True, batch=8, datasets_dict: dict = None, rank: int = 1, seed: int = 101):
    """通过读取一个配置文件去设置各个数据集的dataloader类的设置

    Parameters
    ----------
        rank
        seed
        config_path: s
        is_train: bool
            默认 TRUE, train_dataset的loader 设置
            否则, test_dataset的loader 设置
        datasets_dict: dict
            由dataset类构成的一个字典. get_datasets()

    Returns
    -------
        字典: dict
            key是数据集名称, value是该数据集dataloader类
    """
    # loader 配置文件
    # config_path = '../config_yaml/dataloader_for_lpips.yaml'
    # with open(config_path, 'r') as stream:
    #     config = ini.yaml.safe_load(stream)

    # train_loader 还是 test_loader??
    dataloaders_dict = {}
    my_batch_size = configs['batch_size']
    is_shuffle = configs['train_shuffle']
    is_drop_last = configs['drop_last']

    if is_train:
        loader_type = 'train'
    else:
        loader_type = 'test'
    is_shuffle = configs['test_shuffle']

    for key, value in datasets_dict.items():  # key 是数据集名称, value是数据集的dataset类
        # my_batch_size = config[loader_type][key]['batch_size']

        dataloaders_dict[key] = ini.DataLoader(dataset=value, batch_size=my_batch_size,
                                               shuffle=is_shuffle,
                                               drop_last=is_drop_last,
                                               num_workers=6,
                                               pin_memory=False,
                                               worker_init_fn=partial(worker_init_fn, rank, seed)

                                               )
    print(f'{loader_type}的dataloader加载完成!')
    print(f'dataloader 有{dataloaders_dict.keys()}')
    return dataloaders_dict


def get_model(model_name: str = ''):
    """给定一个model name 返回该 model

    Parameters
    ----------
        model_name: str
            默认空

    Returns
    -------
        model对象
    """
    if model_name == 'vgg16':
        model = ini.Vgg16Modify()
    elif model_name == 'resnet18':
        model = ini.Resnet18Modify()
    elif model_name == 'resnet34':
        model = ini.Resnet34Modify()
    elif model_name == 'resnet50':
        model = ini.Resnet50Modify()
    elif model_name == 'dists':
        model = DISTS()
    else:
        raise ValueError('给定的model名称错误!!!')

    return model


def parse_run_config_file(config_path: str = '', mode: str = 'base', elem: str = ''):
    """ 解析 run_config_IQA.yaml, 该文件保存一些运行时的配置.

    Parameters
    ----------
        mode: str
            default 'base' 不增强
            'aug' 代表增强
        config_path: str
            配置文件路径
        elem: str
            配置文件中的key
    Returns
    -------
        object: key对应的value
            str, int, 都有可能.
    """

    with open(config_path, 'r') as stream:
        config_dict = ini.yaml.safe_load(stream)
    return config_dict[mode][elem]


def get_loss_fn(model_loss_name: str = 'default'):
    """

    Parameters
    ----------

        model_loss_name: str
            ex: lpips

    Returns
    -------
        loss object
    """

    loss_fn = 0
    if model_loss_name == 'vgg16' or model_loss_name == 'resnet18' or model_loss_name == 'resnet34' or model_loss_name == 'resnet50':
        loss_fn = ini.nn.MSELoss()
    elif model_loss_name == 'dists':
        loss_fn = ini.nn.L1Loss()
    else:
        raise ValueError("没有该model的loss!!!")
    return loss_fn


# def get_optim_and_scheduler(model_name: str = 'default'):
#     if model_name == 'lpips' or model_name == 'vgg16':
#         optim = ini.torch.optim.Adam()
def msg(message: str = 'default') -> object:
    """ 打印时间

    Parameters
    ----------
        message: str
            一个消息字符串

    Returns
    -------
    :rtype: object

    """
    # logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(message)s', datefmt='%m/%d/%y %I:%M:%S %0')
    logging.basicConfig(level=logging.INFO, format='%(message)s %(asctime)s', datefmt='%m/%d %I:%M %p')
    logging.info(message)


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed: int = 101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    ini.np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    ini.np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------#
#   train loop
# ---------------------------------------------------#
def write_csv(path: str, args: list):
    """ 保存到csv文件

    Parameters
    ----------
        path: str
            保存路径
        args: list


    Returns
    -------
        打印一条保存成功信息!
    """
    with open(path, 'a+') as f:
        csvwriter = ini.csv.writer(f)
        csvwriter.writerow(args)
    # print("结果写入csv文件成功!!")


def train_loop(epoch, model, dataloaders_dict: dict,
               loss_fn, optimizer, device, print_iter, dir_path):
    for key, value in dataloaders_dict.items():  # key 是 数据集name, value 是 该数据集的dataloader
        size = len(value.dataset)
        num_batches = len(value)
        print(f'Train: [{key}] num_image: [{size}] num_batches: [{num_batches}]')
        iter_loss = 0  # 累计 loss, 固定iter 打印该信息
        iter_count = 0  # 记录 iter 次数
        for data in value:
            # 注意所有的数据集分数处理过均是 0-1, 并且 越大质量越好
            # 注意numpy参与数据增强, 数据类型是float64, 一般神经网络数据类型是float32.

            #  todo label 没有进行stack
            ref_imgs, dist_imgs, labels = data

            # bs, n_da, c, h, w = ref_imgs.size()  # n_da 扩增倍数
            labels = labels.float()  # float64 -> float32
            ref_imgs, dist_imgs, labels = ref_imgs.to(device), dist_imgs.to(device), labels.to(device)  # CPU->GPU

            # 展开
            # ref_imgs = ref_imgs.view(-1, c, h, w)  # (bs,n_da 3, 256, 256) - > (bs*n_da,3, 256, 256)
            # dist_imgs = dist_imgs.view(-1, c, h, w)
            # labels = labels.view(-1, 1)
            if isinstance(model, DISTS):  # 单独处理DISTS
                pred_score = model.forward(ref_imgs, dist_imgs, True)
            else:
                pred_score = model.forward(ref_imgs, dist_imgs)
            # pred_score size : [batch_size, 1]
            # But labels size : [batch_size]
            pred_score = pred_score.squeeze()
            labels = labels.squeeze()
            loss = loss_fn(pred_score, labels)
            iter_loss += loss.item()  # 累计 print_iter 个 iter的loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_count += 1
            # 每print_iter 个保存 打印结果
            if iter_count % print_iter == 0:
                avg_loss = iter_loss / iter_count
                print(f'loss: {avg_loss:.8f}  iter: [{iter_count}/{num_batches}]')
                # 保存结果
                result_list = [epoch, iter_count, avg_loss]
                write_csv(dir_path + '/train.csv', result_list)

        return iter_loss / iter_count  # 返回训练平均loss


def train_loop2(epoch, model, dataloaders_dict: dict,
                loss_fn, optimizer, device, print_iter, dir_path):
    iter_loss = 0  # 累计 loss, 固定iter 打印该信息
    iter_count = 0  # 记录 iter 次数
    for key, value in dataloaders_dict.items():  # key 是 数据集name, value 是 该数据集的dataloader
        for data in dataloaders_dict['tid2013_50_08']:

            # 注意所有的数据集分数处理过均是 0-1, 并且 越大质量越好
            # 注意numpy参与数据增强, 数据类型是float64, 一般神经网络数据类型是float32.

            ref_imgs, dist_imgs, labels = data

            labels = labels.float()  # float64 -> float32

            ref_imgs, dist_imgs, labels = ref_imgs.to(device), dist_imgs.to(device), labels.to(device)  # CPU->GPU
            if isinstance(model, DISTS):  # 单独处理DISTS
                pred_score = model.forward(ref_imgs, dist_imgs, True)
            else:
                pred_score = model.forward(ref_imgs, dist_imgs)
            # pred_score size : [batch_size, 1]
            # But labels size : [batch_size]
            pred_score = pred_score.squeeze()
            loss = loss_fn(pred_score, labels)
            iter_loss += loss.item()  # 累计 print_iter 个 iter的loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_count += 1
            # 每print_iter 个保存 打印结果
            if iter_count % print_iter == 0:
                avg_loss = iter_loss / iter_count
                msg('\n')
                print(f'||Train epoch: {epoch} ||iter: {iter_count} ||')
                print(f'||平均loss是{avg_loss:.5f} ||')
                # 保存结果
                result_list = [epoch, iter_count, avg_loss]
                write_csv(dir_path + '/train.csv', result_list)

        return iter_loss / iter_count  # 返回训练平均loss


# ---------------------------------------------------#
#   test loop
# ---------------------------------------------------#
def test_loop(epoch, model, dataloaders_dict: dict,
              loss_fn, device):
    model.eval()
    result_list = []  # 记录所有的 plcc srcc krcc loss
    psk = 0  # 测试结果中的 plcc + srcc + krcc
    ps = 0  # 测试结果中的 plcc + srcc
    with torch.no_grad():
        for key, value in (dataloaders_dict.items()):
            size = len(value.dataset)
            num_batches = len(value)
            print(f'Test: [{key}] num_image: [{size}] num_batches: [{num_batches}]')
            # key 是 数据集名称, value是数据集的dataloader 类
            test_all_loss = 0
            batch = 0
            label_pred_all_batch = 0  # 记录预测的pred
            labels_all_batch = 0  # 记录真实的pred
            for data in value:
                batch += 1  # 累计 iter 次数
                # 注意所有的数据集分数处理过均是 0-1, 并且 越大质量越好
                # 注意numpy参与数据增强, 数据类型是float64, 一般神经网络数据类型是float32.
                ref_imgs, dist_imgs, labels = data

                # bs, n_da, c, h, w = ref_imgs.size()
                # labes = labels.float()
                ref_imgs, dist_imgs, labels = ref_imgs.to(device), dist_imgs.to(device), labels.to(device)
                # CPU->GPU

                # ref_imgs = ref_imgs.view(-1, c, h, w)  # (bs,n_da 3, 256, 256) - > (bs*n_da,3, 256, 256)
                # dist_imgs = dist_imgs.view(-1, c, h, w)
                # labels = labels.view(-1, 1)  # （bs*n_da,1）

                pred_score = model.forward(ref_imgs, dist_imgs)
                pred_score = pred_score.squeeze()
                labels = labels.squeeze()  # (32, 1) -> (32)
                test_all_loss = test_all_loss + loss_fn(pred_score, labels).item()  # 累加 每个batch的loss

                # 所有batch结果concat
                if batch == 1:
                    # shape: (batch_size,)
                    label_pred_all_batch = pred_score
                    labels_all_batch = labels

                else:
                    label_pred_all_batch = torch.cat([label_pred_all_batch, pred_score])
                    labels_all_batch = torch.cat([labels_all_batch, labels])

            labels_pred_all = label_pred_all_batch.cpu().numpy()
            # print(labels_pred_all.shape)
            labels_all = labels_all_batch.cpu().numpy()

            # -------------------------------
            # 计算 PLCC SR0CC KR0CC
            # -------------------------------
            plcc, _ = ini.stats.pearsonr(labels_pred_all, labels_all)
            # plcc = ini.hold_decimal_point(plcc, 4)  # 保留小数点后4位 TODO
            srcc, _ = ini.stats.spearmanr(labels_pred_all, labels_all)
            # srcc = ini.hold_decimal_point(srcc, 4)  # TODO
            krcc, _ = ini.stats.kendalltau(labels_pred_all, labels_all)
            # krcc = ini.hold_decimal_point(krcc, 4)  # TODO
            test_loss_avg = test_all_loss / batch
            # test_loss_avg = ini.hold_decimal_point(test_loss_avg, 8)  # TODO
            psk = psk + plcc + srcc + krcc
            ps = ps + plcc + srcc
            result_list.append(plcc)
            result_list.append(srcc)
            result_list.append(krcc)
            result_list.append(test_loss_avg)
        return result_list, psk, ps


def test_loop2(epoch, model, dataloaders_dict: dict,
               loss_fn, device):
    model.eval()
    result_list = []  # 记录所有的 plcc srcc krcc loss
    psk = 0  # 测试结果中的 plcc + srcc + krcc
    ps = 0  # 测试结果中的 plcc + srcc
    with torch.no_grad():
        for key, value in (dataloaders_dict.items()):  # key 是 数据集名称, value是数据集的dataloader 类
            test_all_loss = 0
            batch = 0
            label_pred_all_batch = 0  # 记录预测的pred
            labels_all_batch = 0  # 记录真实的pred
            import time  #
            # print(f"start: {time.ctime()}")
            for data in value:

                # print(f"end:  {time.ctime()}")

                batch += 1  # 累计 iter 次数
                # 注意所有的数据集分数处理过均是 0-1, 并且 越大质量越好
                # 注意numpy参与数据增强, 数据类型是float64, 一般神经网络数据类型是float32.
                ref_imgs, dist_imgs, labels = data
                ref_imgs, dist_imgs, labels = ref_imgs.to(device), dist_imgs.to(device), labels.to(device)  # CPU->GPU
                pred_score = model.forward(ref_imgs, dist_imgs)
                pred_score = pred_score.squeeze()  # (32, 1) -> (32)
                test_all_loss = test_all_loss + loss_fn(pred_score.squeeze(), labels).item()  # 累加 每个batch的loss

                # 所有batch结果concat
                if batch == 1:
                    # shape: (batch_size,)
                    label_pred_all_batch = pred_score
                    labels_all_batch = labels

                else:
                    label_pred_all_batch = torch.cat([label_pred_all_batch, pred_score])
                    labels_all_batch = torch.cat([labels_all_batch, labels])

            labels_pred_all = label_pred_all_batch.cpu().numpy()
            print(labels_pred_all.shape)
            labels_all = labels_all_batch.cpu().numpy()

            # -------------------------------
            # 计算 PLCC SR0CC KR0CC
            # -------------------------------
            plcc, _ = ini.stats.pearsonr(labels_pred_all, labels_all)
            # plcc = ini.hold_decimal_point(plcc, 4)  # 保留小数点后4位 TODO
            srcc, _ = ini.stats.spearmanr(labels_pred_all, labels_all)
            # srcc = ini.hold_decimal_point(srcc, 4)  # TODO
            krcc, _ = ini.stats.kendalltau(labels_pred_all, labels_all)
            # krcc = ini.hold_decimal_point(krcc, 4)  # TODO

        return plcc, srcc, krcc


# n倍扩增实验test 测试 取均值
def test_loop3(epoch, model, dataloaders_dict: dict,
               loss_fn, device):
    model.eval()
    result_list = []  # 记录所有的 plcc srcc krcc loss
    psk = 0  # 测试结果中的 plcc + srcc + krcc
    ps = 0  # 测试结果中的 plcc + srcc
    with torch.no_grad():
        for key, value in (dataloaders_dict.items()):
            size = len(value.dataset)
            num_batches = len(value)
            print(f'Test: [{key}] num_image: [{size}] num_batches: [{num_batches}]')
            # key 是 数据集名称, value是数据集的dataloader 类
            test_all_loss = 0
            batch = 0
            label_pred_all_batch = 0  # 记录预测的pred
            labels_all_batch = 0  # 记录真实的pred
            for data in value:
                batch += 1  # 累计 iter 次数
                # 注意所有的数据集分数处理过均是 0-1, 并且 越大质量越好
                # 注意numpy参与数据增强, 数据类型是float64, 一般神经网络数据类型是float32.
                ref_imgs, dist_imgs, labels = data

                bs, n_da, c, h, w = ref_imgs.size()
                labes = labels.float()
                ref_imgs, dist_imgs, labels = ref_imgs.to(device), dist_imgs.to(device), labels.to(device)
                # CPU->GPU

                ref_imgs = ref_imgs.view(-1, c, h, w)  # (bs,n_da 3, 256, 256) - > (bs*n_da,3, 256, 256)
                dist_imgs = dist_imgs.view(-1, c, h, w)
                labels = labels.view(-1, 1)  # (bs, n_da) ->（bs*n_da,1）

                pred_score = model.forward(ref_imgs, dist_imgs)  # (bs*n_da, 1)

                # 对扩增n_da个图像的预测结果取均值
                pred_score = pred_score.view(bs, n_da, -1).mean(1)  # (bs*n_da, 1)->(bs,n_da, -1)->(bs, 1)
                labels = labels.view(bs, n_da, -1).mean(1)      # -> -> (bs,1)

                pred_score = pred_score.squeeze()
                labels = labels.squeeze()
                test_all_loss = test_all_loss + loss_fn(pred_score, labels).item()  # 累加 每个batch的loss

                # 所有batch结果concat
                if batch == 1:
                    # shape: (batch_size,)
                    label_pred_all_batch = pred_score
                    labels_all_batch = labels

                else:
                    label_pred_all_batch = torch.cat([label_pred_all_batch, pred_score])
                    labels_all_batch = torch.cat([labels_all_batch, labels])

            labels_pred_all = label_pred_all_batch.cpu().numpy()
            # print(labels_pred_all.shape)
            labels_all = labels_all_batch.cpu().numpy()

            # -------------------------------
            # 计算 PLCC SR0CC KR0CC
            # -------------------------------
            plcc, _ = ini.stats.pearsonr(labels_pred_all, labels_all)
            # plcc = ini.hold_decimal_point(plcc, 4)  # 保留小数点后4位 TODO
            srcc, _ = ini.stats.spearmanr(labels_pred_all, labels_all)
            # srcc = ini.hold_decimal_point(srcc, 4)  # TODO
            krcc, _ = ini.stats.kendalltau(labels_pred_all, labels_all)
            # krcc = ini.hold_decimal_point(krcc, 4)  # TODO
            test_loss_avg = test_all_loss / batch
            # test_loss_avg = ini.hold_decimal_point(test_loss_avg, 8)  # TODO
            psk = psk + plcc + srcc + krcc
            ps = ps + plcc + srcc
            result_list.append(plcc)
            result_list.append(srcc)
            result_list.append(krcc)
            result_list.append(test_loss_avg)
        return result_list, psk, ps


def hold_decimal_point(number: float, retain: int = 4):
    """ 将浮点类型保留自己想要的位数.

    Parameters
    ----------
        number: float
            传入一个浮点类型小数
        retain: int
            保留小数点后的位数

    Returns
    -------
        float类型
    """
    number = f'{number:.{retain}f}'
    return float(number)


def create_dir(time_seq_str, *dir_path):
    import os
    for count, elem in enumerate(dir_path):
        elem = os.path.join(elem, time_seq_str)
        if os.path.exists(elem) is False:
            os.makedirs(elem)
            print(f"路径{elem}创建成功!")
        else:
            print(f"{elem}已经存在")


def get_time(fmt='YMDHM'):
    """ 获取时间

    :param fmt: str
            默认 YMDHM: 代表年月日时分
    :return:
            期望的时间字符串
    """
    sec = time.time()
    result = time.localtime(sec)
    if fmt == 'YMDHM':
        time_str = str(result.tm_year) + str(result.tm_mon) + str(result.tm_mday) + str(result.tm_hour) + str(
            result.tm_min) + str(result.tm_sec)

    return time_str


def get_run_config_path(model: str = 'vgg16', database_name='tid2013_08'):
    """ 给定一个model name 返回其run_config_IQA.yaml文件相对地址

    Parameters
    ----------
        model: str
            模型名称
        database_name:  str, 默认 tid2013_08, 100%大小tid2013, 再八二划分训练测试集
                        tid2013_50_08 50%tid2013,再 八二划分训练测试
            训练集和测试集, 主要是针对vgg16等backbone, 会在不同的数据集上划分数据集进行训练测试


    Returns
    -------
        path1: str
            dataloader_for_lpips.yaml
        path2: str
            run_config_IQA.yaml 文件相对地址


    """
    path1 = 0  # dataloader bath_size 大小 配置文件地址
    path2 = 0  # 运行模型的一些参数配置文件地址
    # vgg16 的配置文件
    if model == 'vgg16':
        if database_name == 'tid2013_08':  # 在 100% 大小的tid2013上 80% 测试 20%训练
            path1 = '../config_yaml/dataloader_for_vgg16.yaml'
            path2 = '../config_yaml/run_config_vgg16.yaml'
        elif database_name == 'kadid10k_08':  # 在 100%大小的kadid10k 80%测试, 20%训练
            path1 = '../config_yaml/dataloader2_for_vgg16.yaml'
            path2 = '../config_yaml/run_config2_vgg16.yaml'
        elif database_name == 'tid2013_50_08':  # 在 50%大小的tid2013 80%测试, 20%训练
            path1 = '../config_yaml/dataloader2_for_vgg16.yaml'
            path2 = '../config_yaml/run_config2_vgg16.yaml'

    elif model == 'resnet18':
        if database_name == 'tid2013_08':  # 在 tid2013上 80% 测试 20%训练
            path1 = '../config_yaml/dataloader_for_vgg16.yaml'
            path2 = '../config_yaml/run_config_resnet18.yaml'
        elif database_name == 'kadid10k_08':  # 在kadid10k 80%测试, 20%训练
            path1 = '../config_yaml/dataloader2_for_vgg16.yaml'
            path2 = '../config_yaml/run_config2_resnet18.yaml'
    elif model == 'resnet34':
        if database_name == 'tid2013_08':  # 在 tid2013上 80% 测试 20%训练
            path1 = '../config_yaml/dataloader_for_vgg16.yaml'
            path2 = '../config_yaml/run_config_resnet34.yaml'
        elif database_name == 'kadid10k_08':  # 在kadid10k 80%测试, 20%训练
            path1 = '../config_yaml/dataloader2_for_vgg16.yaml'
            path2 = '../config_yaml/run_config2_resnet34.yaml'
    elif model == 'dists':
        if database_name == 'tid2013_08':  # 在 tid2013上 80% 测试 20%训练
            path1 = '../config_yaml/dataloader_for_vgg16.yaml'
            path2 = '../config_yaml/run_config_dists.yaml'
        elif database_name == 'kadid10k_08':  # 在kadid10k 80%测试, 20%训练
            path1 = '../config_yaml/dataloader2_for_vgg16.yaml'
            path2 = '../config_yaml/run_config2_dists.yaml'
        elif database_name == 'kadid10k':  # kadid10k跨数据集测试    # TODO 跨数据集配置文件添加待完成
            pass
    else:  # TODO 更多配置文件需要添加, DeepWSD
        pass

    return path1, path2


def froze_layer(epoch, froze_epoch: int = 1, model=None, model_name: str = 'vgg16'):
    """ 冻结网络部分层  # TODO 添加新的model 需要改动这里!!!
        epoch: int 当前epoch
        froze_epoch: int, 设定冻结epoch
        model: nn.Moudle
        model_name: str, 模型名
    """
    if model_name == 'vgg16':
        if epoch <= froze_epoch:
            # froze_epoch内冻结 特征提取层, 只训练FC层
            for param in model.stage1.parameters():
                param.requires_grad = False
            for param in model.stage2.parameters():
                param.requires_grad = False
        else:
            # 大于 froze_epoch后, 解冻特征提取层, 整个网络训练
            for param in model.stage1.parameters():
                param.requires_grad = True
            for param in model.stage2.parameters():
                param.requires_grad = True
    elif model_name == 'resnet18' or model_name == 'resnet34' or model_name == 'resnet50':
        if epoch <= froze_epoch:
            # froze_epoch内冻结 特征提取层, 只训练FC层
            for param in model.stage.parameters():
                param.requires_grad = False

        else:
            # 大于 froze_epoch后, 解冻特征提取层, 整个网络训练
            for param in model.stage.parameters():
                param.requires_grad = True
    else:
        raise AssertionError('没有该模型冻结处理方式!!')


def get_optim(model: str = 'vgg16', mode: str = 'base'):
    # if model == 'vgg16':
    #     # if mode == 'base':
    #     #     optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
    pass


if __name__ == '__main__':
    # kadid10k_50_08 = get_datasets(True,transforms, 'base', 'kadidk-50-08' )
    # result = get_datasets(True, None, 'live', 'csiq')
    # # result2 = get_datasets(True, None, '')
    #     # # get_model()
    #     # # m = '||srcc : 0546|| plcc:54545||'
    #     # # msg(m)
    #     # # # import logging
    #     # # # logging.basicConfig(level=logging.INFO)
    #     # # # logging.info('test')
    #     # # num = hold_decimal_point(0.773446, 5)
    #     # # print(num)
    #     # # path = 'm.csv'
    #     # # a = [0.545, 0.3434, 0.3434]
    #     # # write_csv(path, a)
    #     # model_save_dir = '../checkpoint/vgg16/base/1016'
    #     # create_dir(model_save_dir)
    #     print(get_time())
    msg('kaishi')
