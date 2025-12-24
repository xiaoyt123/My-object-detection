import argparse
import math
import os
import time
from glob import glob

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import tomllib
import numpy as np
import cv2
from tqdm import tqdm
from utils.datasets import LoadImagesAndLabels

# 第三方库与自定义模块导入
import test  # import test.py to get mAP after each epoch
from models import (
    Darknet, F, FocalLoss, MemoryEfficientSwish, Mish, ONNX_EXPORT, Path, Swish,
    SwishImplementation, YOLOLayer, ap_per_class, apply_classifier, attempt_download,
    bbox_iou, box_iou, build_targets, clip_coords, coco80_to_coco91_class, coco_class_count,
    coco_class_weights, coco_only_people, coco_single_class_labels, compute_ap, compute_loss,
    convert, create_backbone, create_grids, create_modules, crop_images_random,
    download_blob, fitness, gdrive_download, get_yolo_layers, init_seeds, kmean_anchors,
    labels_to_class_weights, labels_to_image_weights, load_classes, load_darknet_weights,
    matplotlib, nn, non_max_suppression, parse_data_cfg, parse_model_cfg,
    plot_evolution_results, plot_images, plot_one_box, plot_results, plot_results_overlay,
    plot_targets_txt, plot_test_txt, plot_wh_methods, plt, print_model_biases,
    print_mutation, random, save_weights, scale_coords, select_best_evolve, shutil,
    smooth_BCE, strip_optimizer, torch_utils, torchvision, upload_blob,
    weightedFeatureFusion, weights_init_normal, wh_iou, xywh2xyxy, xyxy2xywh
)

# 全局配置
mixed_precision = False
wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# 读取超参数配置
with open('My-object-detection/yolo/hyp.toml', 'rb') as file:
    hyp = tomllib.load(file)

print(f"初始学习率: {hyp['lr']['lr0']}")
print(f"分类损失权重: {hyp['loss']['cls']}")


def train(opt, device):
    """
    训练函数（修复版）
    :param opt: 命令行参数对象
    :param device: 训练设备 (cpu/cuda)
    """
    # 基础参数配置
    cfg = opt.cfg
    data = opt.data
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate
    weights = opt.weights

    # 初始化随机种子
    init_seeds()

    # 解析数据配置文件
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # 类别数

    # 删除历史结果文件
    for f in glob('*_batch*.jpg') + glob(results_file):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    # 创建权重目录
    os.makedirs(wdir, exist_ok=True)

    # 初始化模型并加载到设备
    model = Darknet(cfg).to(device)

    # 优化器参数组配置（修复核心：完整参数组）
    pg0, pg1, pg2 = [], [], []  # 分别对应：普通参数、卷积权重、偏置
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # 偏置参数
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # 卷积权重（应用权重衰减）
        else:
            pg0.append(v)  # 其他参数

    # 优化器选择（Adam/SGD）
    weight_decay = hyp.get('weight_decay', 5e-4)
    momentum = hyp.get('momentum', 0.937)
    lr0 = hyp['lr']['lr0']

    if opt.adam:
        optimizer = optim.Adam([
            {'params': pg0, 'lr': lr0},
            {'params': pg1, 'lr': lr0, 'weight_decay': weight_decay},
            {'params': pg2, 'lr': lr0 * 2}  # 偏置学习率翻倍
        ], lr=lr0)
    else:
        optimizer = optim.SGD([
            {'params': pg0, 'lr': lr0, 'momentum': momentum},
            {'params': pg1, 'lr': lr0, 'momentum': momentum, 'weight_decay': weight_decay},
            {'params': pg2, 'lr': lr0 * 2, 'momentum': momentum}
        ], lr=lr0)

    # 训练起始参数
    start_epoch = 0
    best_fitness = 0.0

    # 加载预训练权重
    if len(weights) > 0 and os.path.exists(weights):
        load_darknet_weights(model, weights)
        print(f"成功加载预训练权重: {weights}")

    # 学习率调度器（余弦退火）
    lf = lambda x: (1 + math.cos(x * math.pi / epochs)) / 2  # 余弦退火公式
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # 数据集加载（核心修复：禁用多线程 num_workers=0）
    dataset = LoadImagesAndLabels(
        train_path, img_size, batch_size,
        augment=True,
        hyp=hyp,
        rect=opt.rect,
        cache_images=opt.cache_images,
        single_cls=opt.single_cls
    )

    # 数据加载器（禁用多线程避免内存错误）
    nw = 0  # 关键修复：num_workers=0
    batch_size = min(batch_size, len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=not opt.rect,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # 测试集加载器（同样禁用多线程）
    testloader = torch.utils.data.DataLoader(
        LoadImagesAndLabels(
            test_path, img_size_test, batch_size * 2,
            hyp=hyp,
            rect=True,
            cache_images=opt.cache_images,
            single_cls=opt.single_cls
        ),
        batch_size=batch_size * 2,
        num_workers=nw,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # 模型附加参数
    model.nc = nc
    model.hyp = hyp
    model.gr = 0.0  # giou loss ratio
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)

    # 训练状态初始化
    nb = len(dataloader)  # 批次总数
    prebias = start_epoch == 0
    maps = np.zeros(nc)  # 每类mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP, F1, val GIoU, val Objectness, val Classification
    t0 = time.time()

    # 开始训练
    print(f'使用 {nw} 个数据加载线程')
    print(f'开始训练 {epochs} 轮，批次大小: {batch_size}')

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(4).to(device)  # 平均损失 (GIoU, obj, cls, total)
        pbar = tqdm(enumerate(dataloader), total=nb, desc=f'Epoch {epoch + 1}/{epochs}')

        # Prebias 阶段（前3轮仅训练偏置）
        if prebias:
            ne = 3
            ps = (0.1, 0.9) if epoch < ne else (lr0, momentum)
            if epoch == ne:
                model.gr = 1.0  # 启用giou loss
                print_model_biases(model)
                prebias = False

            # 更新偏置学习率
            for param_group in optimizer.param_groups:
                if param_group['params'] == pg2:
                    param_group['lr'] = ps[0]
                    if 'momentum' in param_group:
                        param_group['momentum'] = ps[1]

        # 批次循环
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # 累计批次号
            imgs = imgs.to(device).float() / 255.0  # 归一化到0-1
            targets = targets.to(device)

            # Burn-in 阶段（前200批次）
            n_burn = 200
            if ni <= n_burn:
                # 延迟BatchNorm统计更新
                for _, m in model.named_modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = ni == n_burn

            # 前向传播
            pred = model(imgs)

            # 计算损失
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print(f'警告：非有限损失值，终止训练 | 损失值: {loss_items}')
                return results

            # 损失缩放（适配标称批次64）
            loss *= batch_size / 64

            # 反向传播
            loss.backward()

            # 累计梯度更新
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 更新平均损失
            mloss = (mloss * i + loss_items) / (i + 1)

            # 进度条显示
            mem = f"{torch.cuda.memory_allocated() / 1E9:.3f}G" if torch.cuda.is_available() else "0G"
            pbar.set_postfix({
                'gpu_mem': mem,
                'GIoU': f'{mloss[0]:.3f}',
                'obj': f'{mloss[1]:.3f}',
                'cls': f'{mloss[2]:.3f}',
                'total': f'{mloss[3]:.3f}',
                'targets': len(targets),
                'img_size': img_size
            })

            # 内存清理（核心修复：避免张量堆积）
            del imgs, targets, pred, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 更新学习率
        scheduler.step()

        # 验证阶段（计算mAP）
        if not opt.notest or (epoch + 1 == epochs):
            is_coco = any(x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']) and nc == 80
            results, maps = test.test(
                cfg, data,
                batch_size=batch_size * 2,
                img_size=img_size_test,
                model=model,
                conf_thres=0.001 if (epoch + 1 == epochs) else 0.01,
                iou_thres=0.6,
                save_json=(epoch + 1 == epochs) and is_coco,
                single_cls=opt.single_cls,
                dataloader=testloader
            )

        # 保存结果到文件
        with open(results_file, 'a') as f:
            f.write(f"{epoch + 1}/{epochs}\t{mem}\t{mloss[0]:.3f}\t{mloss[1]:.3f}\t{mloss[2]:.3f}\t{mloss[3]:.3f}\t"
                    f"{len(targets)}\t{img_size}\t{results[0]:.3f}\t{results[1]:.3f}\t{results[2]:.3f}\t{results[3]:.3f}\t"
                    f"{results[4]:.3f}\t{results[5]:.3f}\t{results[6]:.3f}\n")

        # 更新最佳模型
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi

        # 保存权重
        if not opt.nosave or (epoch + 1 == epochs):
            # 构建检查点
            chkpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': open(results_file, 'r').read(),
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': None if (epoch + 1 == epochs) else optimizer.state_dict()
            }

            # 保存最后一轮权重
            torch.save(chkpt, last)

            # 保存最佳权重
            if fi == best_fitness:
                torch.save(chkpt, best)

            del chkpt

        # 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练结束后重命名结果文件
    if len(opt.name):
        n = '_' + opt.name if not opt.name.isnumeric() else opt.name
        os.rename(results_file, f'results{n}.txt')
        if os.path.exists(last):
            os.rename(last, wdir + f'last{n}.pt')
        if os.path.exists(best):
            os.rename(best, wdir + f'best{n}.pt')

    # 绘制结果图
    if not opt.evolve:
        plot_results()

    # 打印训练耗时
    total_time = (time.time() - t0) / 3600
    print(f'\n训练完成！共 {epochs} 轮，耗时 {total_time:.3f} 小时')

    # 清理分布式训练（如果启用）
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return results



if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='My-object-detection/yolo/args.toml',
                        help='TOML配置文件路径')
    parser.add_argument('--override', nargs='*', help='覆盖配置参数，格式: key=value')
    args = parser.parse_args()

    # 加载TOML配置文件
    with open(args.config, 'rb') as f:
        config = tomllib.load(f)

    # 创建参数对象
    class Opt:
        pass

    opt = Opt()

    # 从配置文件赋值参数
    opt.epochs = config['training']['epochs']
    opt.batch_size = config['training']['batch_size']
    opt.accumulate = config['training']['accumulate']
    opt.cfg = config['paths']['config']
    opt.data = config['paths']['data']
    opt.multi_scale = config['training']['multi_scale']
    opt.img_size = config['image_settings']['img_size']
    opt.rect = config['training']['rectangular']
    opt.resume = config['training']['resume']
    opt.nosave = config['training']['nosave']
    opt.notest = config['training']['notest']
    opt.evolve = config['training']['evolve']
    opt.bucket = config['paths']['bucket']
    opt.cache_images = config['training']['cache_images']
    opt.weights = config['paths']['weights']
    opt.name = config['paths']['results_name']
    opt.device = config['hardware']['device']
    opt.adam = config['training']['adam_optimizer']
    opt.single_cls = config['training']['single_class']
    opt.var = config['debug']['debug_variable']

    # 处理命令行覆盖参数
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            if hasattr(opt, key):
                # 类型转换
                orig_value = getattr(opt, key)
                if isinstance(orig_value, bool):
                    setattr(opt, key, value.lower() in ['true', '1', 'yes'])
                elif isinstance(orig_value, int):
                    setattr(opt, key, int(value))
                elif isinstance(orig_value, float):
                    setattr(opt, key, float(value))
                elif isinstance(orig_value, list):
                    setattr(opt, key, [int(x) for x in value.strip('[]').split(',')])
                else:
                    setattr(opt, key, value)

    # 处理resume参数
    opt.weights = last if opt.resume else opt.weights

    # 打印参数配置
    print("=" * 50)
    print("训练参数配置:")
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("=" * 50)

    # 选择训练设备（CPU/CUDA）
    device = torch_utils.select_device(opt.device, apex=False, batch_size=opt.batch_size)
    print(f"使用训练设备: {device}")

    # 初始化Tensorboard（可选）
    tb_writer = None
    if not opt.evolve:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter()
            print("Tensorboard已启用，可通过 tensorboard --logdir=runs 查看")
        except ImportError:
            print("警告：未安装Tensorboard，跳过可视化")

    # 开始训练
    train(opt, device)

