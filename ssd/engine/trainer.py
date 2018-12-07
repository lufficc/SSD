import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ssd.utils import distributed_util


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = distributed_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def _save_model(logger, model, model_path):
    vgg_model = model
    if isinstance(model, DistributedDataParallel):
        vgg_model = model.module
    vgg_model.save(model_path)
    logger.info("Saved checkpoint to {}".format(model_path))


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             criterion,
             device,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training")
    model.train()
    save_to_disk = distributed_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_training_time = time.time()
    trained_time = 0
    tic = time.time()
    end = time.time()
    for iteration, (images, boxes, labels) in enumerate(data_loader):
        iteration = iteration + 1
        scheduler.step()
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict({'regression_loss': regression_loss, 'classification_loss': classification_loss})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()
        trained_time += time.time() - end
        end = time.time()
        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            logger.info(
                "Iter: {:06d}, Lr: {:.5f}, Cost: {:.2f}s, Eta: {}, ".format(iteration, optimizer.param_groups[0]['lr'],
                                                                            time.time() - tic,
                                                                            str(datetime.timedelta(seconds=eta_seconds))) +
                "Loss: {:.3f}, ".format(losses_reduced.item()) +
                "Regression Loss {:.3f}, ".format(loss_dict_reduced['regression_loss'].item()) +
                "Classification Loss: {:.3f}".format(loss_dict_reduced['classification_loss'].item()))

            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced.item(), global_step=global_step)
                summary_writer.add_scalar('losses/location_loss', loss_dict_reduced['regression_loss'].item(), global_step=global_step)
                summary_writer.add_scalar('losses/class_loss', loss_dict_reduced['classification_loss'].item(), global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            tic = time.time()

        if save_to_disk and iteration % args.save_step == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_iteration_{:06d}.pth".format(cfg.INPUT.IMAGE_SIZE, iteration))
            _save_model(logger, model, model_path)

    if save_to_disk:
        model_path = os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_final.pth".format(cfg.INPUT.IMAGE_SIZE))
        _save_model(logger, model, model_path)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
