#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: HCN
@time: 2023/10/12 0012 16:09
@work unit:SDNU
@email: dopaminesdnu@163.com
@describe:  训练以及测试的入口
"""

import sys
import os

# import torch.optim.lr_scheduler

# print(os.path.dirname(os.getcwd()))
# # print(sys.path)
sys.path.append(os.path.dirname(os.getcwd()))
# # print(sys.path)

from ini import *  # 项目根路径加入sys.path

# model_name = input("请输入模型名称:\n")
# mode = input("是否进行增强? base or aug?\n")
# model_name = 'resnet18'
# # mode = 'aug'
# database_name = 'tid2013_08'
# args #
# config path # 增加新模型 改1 根据模型名称和数据集添加配置文件 #
# parser
"""必须要cmd时设定 model, mode, gpu 参数"""
parser = argparse.ArgumentParser(description='训练一些设置')
parser.add_argument('-model', default='resnet34', type=str, help='指定model')
parser.add_argument('-database', default='kadid10k_25_08', type=str, help='指定训练集')
parser.add_argument('-mode', default='aug', type=str, help='是否是增强? base or aug')
parser.add_argument('-gpu', default=1, type=int, help='指定gpu')
parser.add_argument('-Tmax', default=40, type=int, help='学习率调整周期')

args = parser.parse_args()

config_path1, config_path2 = get_run_config_path(args.model, args.database)
config_path1 = '../config_yaml/dataloader-k-25-batch128.yaml'  # dataloader config
config_path2 = '../config_yaml/run_config_resnet34_25_k_aug_v1.yaml'  # run config
# print  start msg #
message = f'|| 模型名称: {args.model} || 是否增强?: {args.mode} || 使用GPU: {args.gpu} ||\n配置文件1路径: {config_path1}\n配置文件2路径: {config_path2}'
msg(message)

# random seed 一个是全局种子, 一个是dataloader的种子 #
seed = prcf(config_path2, mode=args.mode, elem='seed')
rank = prcf(config_path2, mode=args.mode, elem='rank')  # dataloader  seed + rank
seed_everything(seed)
device = torch.device('cuda:{}'.format(args.gpu))  # device

# dataset and dataloader #

# get transforms for need
transforms = get_transforms(args.model)  # 增加新模型 改2根据模型设置不同的transforms #
# get all datasets
train_database_name = prcf(config_path2, mode=args.mode, elem='train_database_name')  # 根据配置文件
test_database_name = prcf(config_path2, mode=args.mode, elem='test_database_name')
train_datasets_dict = get_datasets(True, transforms, args.mode, device, train_database_name, )
test_datasets_dict = get_datasets(False, transforms, args.mode, device, test_database_name, )
# get all dataloader
train_dataloaders_dict = get_dataloaders(config_path1, True, train_datasets_dict, rank, seed)
test_dataloaders_dict = get_dataloaders(config_path1, False, test_datasets_dict, rank, seed)

# model #
model = get_model(args.model)
model = model.to(device)

# lr, loss, scheduler #
init_lr = prcf(config_path2, mode=args.mode, elem='init_lr')
weight_decay = prcf(config_path2, mode=args.mode, elem='weight_decay')
loss_fn = get_loss_fn(args.model)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)  # 改动

# epoch loop #

run_epoch = prcf(config_path2, mode=args.mode, elem='run_epoch')  # 训练epoch
print_iter = prcf(config_path2, mode=args.mode, elem='print_msg_iter')  # 多少iter 打印结果

# 获得当前时间,作为目录一部分
time_seq_str = get_time()  # 获取当前时间字符串 ex: 20231114213
train_result_dir = prcf(config_path2, mode=args.mode, elem='train_result_dir')  # 结果保存目录
test_result_dir = prcf(config_path2, mode=args.mode, elem='test_result_dir')  # 结果保存目录
model_save_dir = prcf(config_path2, mode=args.mode, elem='model_save_dir')  # 模型参数保存目录

create_dir(time_seq_str, train_result_dir, model_save_dir)
train_result_dir = os.path.join(train_result_dir, time_seq_str)
test_result_dir = train_result_dir
model_save_dir = os.path.join(model_save_dir, time_seq_str)
# 一些参数设定
best_epoch = 0  # 记录做好结果的epoch
best_result_list = 0  # 记录最好结果
best_psk = 0  # 记录最好的psk(plcc+srcc+krcc)
best_ps = 0  # 记录做好的ps, 以ps结果有提升为主
epoch = 0
# for epoch in tqdm(range(run_epoch), desc='epoch进度', colour='green'):
while True:
    print('\n')
    msg(f'')
    # 冻结处理
    froze_layer(epoch + 1, froze_epoch=0, model=model, model_name=args.model)

    train_loss = train_loop(epoch + 1, model, train_dataloaders_dict, loss_fn, optimizer,
                            device, print_iter, train_result_dir)
    # 测试 #
    result_list, psk, ps = test_loop(epoch + 1, model, test_dataloaders_dict, loss_fn, device)
    write_csv(test_result_dir + '/test.csv', result_list)  # 保存结果
    # 输出本次测试结果
    print('---------------------当前结果--------------------------')
    print('|\tepoch\t|\t\tps\t\t|\t\tpsk\t\t|')
    print('-----------------------------------------------------')
    print(f'|\t{epoch + 1}\t\t|\t{ps}\t|\t{psk}\t|')
    print('-----------------------------------------------------')
    # -------------------------------------------------------------------------
    #  若产生最优结果:
    #       1.记录的最优ps值小于当前测试的ps, 更新best_ps, best_psk, 记录epoch,
    #         打印 best_ps, besk_psk, 以及具体的 plcc srcc krcc
    #       2.保存模型结果
    # ------------------------------------------------------------------------y-
    if best_ps < ps:
        best_ps = ps
        best_psk = psk
        best_epoch = epoch + 1
        # 保存模型
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'best{epoch + 1}.pth'))
    # 输出最优结果
    print('--------------------最优结果---------------------------')
    print('|\tepoch\t|\t\tbest_ps\t\t|\t\tbest_psk\t\t|')
    print('-----------------------------------------------------')
    print(f'|\t{best_epoch}\t\t|\t{best_ps}\t|\t{best_psk}\t|')
    print('-----------------------------------------------------')
    scheduler.step()
    epoch += 1
    # if epoch - best_epoch > args.Tmax:    # TODO 停止条件为一个Tmax# 记录最优结果所在的epoch 与当前运行的epoch之差, 若 超过20就停止运行程序
    #     break
