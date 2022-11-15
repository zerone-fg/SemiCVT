import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
import shutil

from semicvt.dataset.augmentation import generate_unsup_data
from semicvt.dataset.builder import get_loader
from semicvt.model_cnn_trans.model_helper import ModelBuilder
from semicvt.model_cnn_trans.Graphloss import DistillC2C2N
from semicvt.utils.dist_helper import setup_distributed
from semicvt.utils.loss_helper import (
    compute_unsupervised_loss,
    get_criterion,
)
from semicvt.utils.lr_helper import get_optimizer, get_scheduler
from semicvt.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)
from torch import nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)


def drawLosses(Loss_graph, Loss_sup, Loss_Unsup, name):
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(len(Loss_graph)), Loss_graph, label='graph')
    ax1.plot(range(len(Loss_sup)), Loss_sup, label='sup')
    ax1.plot(range(len(Loss_Unsup)), Loss_Unsup, label='Unsup')
    ax1.set_title("Average trainset loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    plt.savefig(os.path.join(cfg["saver"]["snapshot_dir"], name))


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])
    if os.path.exists(cfg["saver"]["snapshot_dir"] + '/mycode') and rank == 0:
        shutil.rmtree(cfg["saver"]["snapshot_dir"] + '/mycode')
    if rank == 0:
        shutil.copytree('../../../../u2pl/models_cnn_trans_light/', cfg["saver"]["snapshot_dir"] + '/mycode/')

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    print("params", sum(p.numel() for p in model.parameters()) / 1e6)

    sup_loss_fn = get_criterion(cfg)
    graph_loss_fn = DistillC2C2N().cuda()
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )
    f = open(os.path.join(cfg["saver"]["snapshot_dir"], 'result.txt'), 'w')
    Loss_graph = []
    Loss_sup = []
    Loss_Unsup = []

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        Loss_graph, Loss_sup, Loss_Unsup = train(
                        model,
                        model_teacher,
                        optimizer,
                        lr_scheduler,
                        sup_loss_fn,
                        graph_loss_fn,
                        train_loader_sup,
                        train_loader_unsup,
                        epoch,
                        tb_logger,
                        logger,
                        Loss_graph,
                        Loss_sup,
                        Loss_Unsup
                    )

        if rank == 0:
            drawLosses(Loss_graph = Loss_graph, Loss_sup=Loss_sup*10, Loss_Unsup=Loss_Unsup*10, name = 'All_loss.png')

        # Validation
        if cfg_trainer["eval_on"] and epoch>=2:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(model, val_loader, epoch, logger)
            else:
                prec = validate(model_teacher, val_loader, epoch, logger)

            f.write('epoch = {0}, mIOU = {1:.5f}\n'.format(epoch, prec))

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    if best_prec != 0:
                        os.remove(
                            osp.join(cfg["saver"]["snapshot_dir"], '%s_%.2f.pth' % (str(best_epoch), best_prec * 100)))
                    best_prec = prec
                    best_epoch = epoch + 1
                    state["best_miou"] = best_prec
                    torch.save(state,
                               osp.join(cfg["saver"]["snapshot_dir"], '%s_%.2f.pth' % (str(best_epoch), best_prec * 100)))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                tb_logger.add_scalar("mIoU val", prec, epoch)
    f.close()

def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    graph_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    Loss_graph,
    Loss_sup,
    Loss_Unsup
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    graph_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _ = loader_u_iter.next()
        image_u = image_u.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            outs_cnn, outs_trans = model(image_l)
            pred_cnn, rep_cnn = outs_cnn["pred"], outs_cnn["rep"]
            pred_large_cnn = F.interpolate(pred_cnn, (h, w), mode="bilinear", align_corners=True)
            pred_trans, rep_trans = outs_trans["pred"], outs_trans["rep"]
            pred_large_trans = F.interpolate(pred_trans, (h, w), mode="bilinear", align_corners=True)

            loss_cnn = sup_loss_fn(pred_large_cnn, label_l)
            loss_trans = sup_loss_fn(pred_large_trans, label_l)
            sup_loss = 0.5 * loss_cnn + 0.5 * loss_trans
            unsup_loss = 0 * rep_cnn.sum() + 0 * rep_trans.sum()
            contra_loss = 0 * rep_cnn.sum() + 0 * rep_trans.sum()
            graph_loss = 0 * rep_cnn.sum() + 0 * rep_trans.sum()

        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            out_u_teacher_cnn, out_u_teacher_trans = model_teacher(image_u)
            pred_u_teacher_cnn, pred_u_teacher_trans = out_u_teacher_cnn["pred"], out_u_teacher_trans["pred"]
            pred_u_teacher_cnn = F.interpolate(
                pred_u_teacher_cnn, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher_trans = F.interpolate(
                pred_u_teacher_trans, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher_cnn = F.softmax(pred_u_teacher_cnn, dim=1)
            pred_u_teacher_trans = F.softmax(pred_u_teacher_trans, dim=1)

            logits_u_aug, label_u_aug = torch.max((pred_u_teacher_cnn + pred_u_teacher_trans)/2, dim=1)
            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )

            else:
                image_u_aug = image_u

            # forward cnn
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs_cnn, outs_trans = model(image_all)
            pred_all_cnn, rep_all_cnn = outs_cnn["pred"], outs_cnn["rep"]
            pred_all_trans, rep_all_trans = outs_trans["pred"], outs_trans["rep"]

            pred_l_cnn, pred_u_cnn = pred_all_cnn[:num_labeled], pred_all_cnn[num_labeled:]
            pred_l_trans, pred_u_trans = pred_all_trans[:num_labeled], pred_all_trans[num_labeled:]
            rep_u_cnn = rep_all_cnn[num_labeled:]
            rep_u_trans = rep_all_trans[num_labeled:]

            pred_l_large_cnn = F.interpolate(
                pred_l_cnn, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_l_large_trans = F.interpolate(
                pred_l_trans, size=(h, w), mode="bilinear", align_corners=True
            )

            # student: supervised loss for labeled
            loss_cnn = sup_loss_fn(pred_l_large_cnn, label_l.clone())
            loss_trans = sup_loss_fn(pred_l_large_trans, label_l.clone())
            sup_loss = 0.5 * loss_cnn + 0.5 * loss_trans

            pred_u_large_trans = F.interpolate(
                pred_u_trans, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large_cnn = F.interpolate(
                pred_u_cnn, size=(h, w), mode="bilinear", align_corners=True
            )

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t_cnn, out_t_trans = model_teacher(image_all)
                pred_all_teacher_cnn, rep_all_teacher_cnn = out_t_cnn["pred"], out_t_cnn["rep"]
                pred_all_teacher_trans, rep_all_teacher_trans = out_t_trans["pred"], out_t_trans["rep"]

                pred_u_teacher_cnn = pred_all_teacher_cnn[num_labeled:]
                pred_u_teacher_trans = pred_all_teacher_trans[num_labeled:]
                rep_u_teacher_cnn = rep_all_teacher_cnn[num_labeled:]
                rep_u_teacher_trans = rep_all_teacher_trans[num_labeled:]

                pred_u_large_teacher_cnn = F.interpolate(
                    pred_u_teacher_cnn, size=(h, w), mode="bilinear", align_corners=True
                )

                pred_u_large_teacher_trans = F.interpolate(
                    pred_u_teacher_trans, size=(h, w), mode="bilinear", align_corners=True
                )

            graph_loss = (0.5 * graph_loss_fn(pred_u_cnn, rep_u_cnn, pred_u_teacher_trans.detach(),
                                              rep_u_teacher_trans.detach()) + \
                          0.5 * graph_loss_fn(pred_u_trans, rep_u_trans, pred_u_teacher_cnn.detach(),
                                              rep_u_teacher_cnn.detach())) * 0.1


            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            unsup_loss_cnn = (
                    compute_unsupervised_loss(
                        pred_u_large_cnn,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher_cnn.detach(),
                    )
                    * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            unsup_loss_trans = (
                    compute_unsupervised_loss(
                        pred_u_large_trans,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher_trans.detach(),
                    )
                    * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            unsup_loss = 0.5 * unsup_loss_cnn + 0.5 * unsup_loss_trans

        loss = sup_loss + unsup_loss + graph_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_graph_loss = graph_loss.clone().detach()
        dist.all_reduce(reduced_graph_loss)
        graph_losses.update(reduced_graph_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "graph {graph_loss.val:.3f} ({graph_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    epoch,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    graph_loss=graph_losses,
                    uns_loss=uns_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("graph Loss", graph_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)

            Loss_graph.append(graph_losses.avg)
            Loss_sup.append(sup_losses.avg)
            Loss_Unsup.append(uns_losses.avg)

    return Loss_graph, Loss_sup, Loss_Unsup


def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter_cnn = AverageMeter()
    union_meter_cnn = AverageMeter()

    intersection_meter_trans = AverageMeter()
    union_meter_trans = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            outs_cnn, outs_trans = model(images)

        # get the output produced by model_teacher
        '''CNN'''
        output_cnn_p = outs_cnn["pred"]
        output_cnn_p = F.interpolate(output_cnn_p, (h, w), mode="bilinear", align_corners=True)
        output_cnn = output_cnn_p.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection_cnn, union_cnn, target_cnn = intersectionAndUnion(
            output_cnn, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection_cnn = torch.from_numpy(intersection_cnn).cuda()
        reduced_union_cnn = torch.from_numpy(union_cnn).cuda()
        reduced_target_cnn = torch.from_numpy(target_cnn).cuda()

        dist.all_reduce(reduced_intersection_cnn)
        dist.all_reduce(reduced_union_cnn)
        dist.all_reduce(reduced_target_cnn)

        intersection_meter_cnn.update(reduced_intersection_cnn.cpu().numpy())
        union_meter_cnn.update(reduced_union_cnn.cpu().numpy())

        '''trans'''
        output_trans_p = outs_trans["pred"]
        output_trans_p = F.interpolate(output_trans_p, (h, w), mode="bilinear", align_corners=True)
        output_trans = output_trans_p.data.max(1)[1].cpu().numpy()

        # start to calculate miou
        intersection_trans, union_trans, target_trans = intersectionAndUnion(
            output_trans, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection_trans = torch.from_numpy(intersection_trans).cuda()
        reduced_union_trans = torch.from_numpy(union_trans).cuda()
        reduced_target_trans = torch.from_numpy(target_trans).cuda()

        dist.all_reduce(reduced_intersection_trans)
        dist.all_reduce(reduced_union_trans)
        dist.all_reduce(reduced_target_trans)

        intersection_meter_trans.update(reduced_intersection_trans.cpu().numpy())
        union_meter_trans.update(reduced_union_trans.cpu().numpy())

        '''mean'''
        output = (output_cnn_p + output_trans_p) / 2
        output = output.data.max(1)[1].cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class_cnn = intersection_meter_cnn.sum / (union_meter_cnn.sum + 1e-10)
    mIoU_cnn = np.mean(iou_class_cnn)

    iou_class_trans = intersection_meter_trans.sum / (union_meter_trans.sum + 1e-10)
    mIoU_trans = np.mean(iou_class_trans)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        logger.info(" * epoch {} cnn {:.2f} trans {:.2f} mean {:.2f}".format(epoch, mIoU_cnn * 100, mIoU_trans * 100, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()