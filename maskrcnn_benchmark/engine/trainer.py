# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import collections

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def write_tf_summary(dictionary, tb_logger, iteration, prefix=''):
    if dist.get_rank() != 0 or tb_logger is None: return
    for k, v in dictionary.items():
        k2 = f'{prefix}/{k}'
        if isinstance(v, collections.Mapping):
            write_tf_summary(v, tb_logger, iteration, prefix=k2)
        else:
            tb_logger.add_scalar(k2, v, iteration)


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    tb_logger,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader.dataset.class_emb_mtx)
    
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        try:
            targets = [target.to(device) for target in targets]
        except:
            pass

        # with torch.autograd.detect_anomaly():
        loss_dict = model(images, targets)

        if isinstance(loss_dict, tuple):
            info_dict, loss_dict = loss_dict
        else:
            info_dict = None

        losses = sum(loss for loss in loss_dict.values())
        losses = losses / float(cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_dict_reduced['loss'] = losses_reduced
        meters.update(**loss_dict_reduced)

        if info_dict is not None:
            info_dict_reduced = reduce_loss_dict(info_dict)
            meters.update(**info_dict_reduced)

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        if iteration % cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS == 0:
            if cfg.SOLVER.CLIP_GRAD_NORM_AT > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_NORM_AT)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.SOLVER.LOG_PERIOD == 0 or iteration == max_iter:
            write_tf_summary(loss_dict_reduced, tb_logger, iteration, prefix='train')
            if info_dict is not None:
                write_tf_summary(info_dict_reduced, tb_logger, iteration, prefix='train')

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            if cfg.TEST.DO_EVAL:
                dl_val_list = make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1))
                for dl, dname in zip(dl_val_list, cfg.DATASETS.TEST):
                    val_results = inference(
                        model,
                        dl,
                        dataset_name=dname,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                    )
                    synchronize()

                    if val_results is not None:
                        val_results, _ = val_results
                        val_results = val_results.results
                        write_tf_summary(val_results, tb_logger, iteration, prefix=f'validation/{dname}')

                if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
                    if module.roi_heads['box'].predictor.embedding_based:
                        module.roi_heads['box'].predictor.set_class_embeddings(
                            data_loader.dataset.class_emb_mtx)

            if not cfg.SOLVER.SKIP_VAL_LOSS:
                with torch.no_grad():
                    if cfg.SOLVER.USE_TRAIN_MODE_FOR_VALIDATION_LOSS:
                        model.train()
                    else:
                        model.eval()
                    for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                        images_val = images_val.to(device)
                        try:
                            targets_val = [target.to(device) for target in targets_val]
                        except:
                            pass
                        loss_dict = model(images_val, targets_val)
                        if isinstance(loss_dict, tuple):
                            info_dict, loss_dict = loss_dict
                        else:
                            info_dict = None
                        losses = sum(loss for loss in loss_dict.values())
                        loss_dict_reduced = reduce_loss_dict(loss_dict)
                        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                        meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                        if info_dict is not None:
                            info_dict_reduced = reduce_loss_dict(info_dict)
                            meters_val.update(**info_dict_reduced)
                logger.info(
                    meters_val.delimiter.join(
                        [
                            "[Validation]: ",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters_val),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                meters_val_dict = {k: v.global_avg for k, v in meters_val.meters.items()}
                write_tf_summary(meters_val_dict, tb_logger, iteration,
                                 prefix=f'validation/{cfg.DATASETS.TEST[0]}')
            model.train()
            synchronize()

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
