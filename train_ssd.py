import argparse
import datetime
import os
import logging
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from eval_ssd import do_evaluation
from ssd.config import cfg
from ssd.datasets import build_dataset
from ssd.module.prior_box import PriorBox
from ssd.utils.misc import str2bool
from ssd.modeling.ssd import MatchPrior
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.modeling.multibox_loss import MultiboxLoss
from ssd.modeling.data_preprocessing import TrainAugmentation


def train(cfg, args):
    summary_writer = None

    if args.use_tensorboard:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    train_transform = TrainAugmentation(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN)
    prior_box = PriorBox(cfg)
    priors = prior_box()
    target_transform = MatchPrior(priors, cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE, cfg.MODEL.THRESHOLD)

    train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform, target_transform=target_transform)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, cfg.SOLVER.BATCH_SIZE, num_workers=4, shuffle=True)

    model = build_ssd_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.train()

    if args.resume:
        logging.info("Resume from the model {}".format(args.resume))
        model.load(args.resume)
    else:
        logging.info("Init from base net {}".format(args.vgg))
        model.init_from_base_net(args.vgg)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    criterion = MultiboxLoss(iou_threshold=cfg.MODEL.THRESHOLD, neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)

    milestones = cfg.SOLVER.LR_STEPS
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.SOLVER.GAMMA)

    epoch_size = len(train_dataset) // cfg.SOLVER.BATCH_SIZE
    max_iter = cfg.SOLVER.MAX_ITER
    # create batch iterator
    batch_iterator = iter(train_loader)
    epoch = 0
    start_training_time = time.time()
    tic = time.time()
    for iteration in range(max_iter):
        if iteration and iteration % epoch_size == 0:
            batch_iterator = iter(train_loader)
            epoch += 1

        scheduler.step()

        cpu_images, boxes, labels = next(batch_iterator)
        images = cpu_images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        if (iteration + 1) % args.log_step == 0:
            logging.info(
                "Iter: {:06d}, Lr: {:.5f}, Cost: {:.2f}s, ".format(iteration + 1, optimizer.param_groups[0]['lr'], time.time() - tic) +
                "Loss: {:.3f}, ".format(loss.item()) +
                "Regression Loss {:.3f}, ".format(regression_loss.item()) +
                "Classification Loss: {:.3f}".format(classification_loss.item()))

            if summary_writer:
                global_step = iteration + 1
                summary_writer.add_scalar('losses/total_loss', loss.item(), global_step=global_step)
                summary_writer.add_scalar('losses/location_loss', loss.item(), global_step=global_step)
                summary_writer.add_scalar('losses/class_loss', loss.item(), global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            tic = time.time()

        if iteration != 0 and (iteration + 1) % args.save_step == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR, "ssd_vgg_iteration_{:06d}.pth".format(iteration + 1))
            model.save(model_path)
            logging.info("Saved checkpoint to {}".format(model_path))
    model_path = os.path.join(cfg.OUTPUT_DIR, "ssd_vgg_final.pth")
    model.save(model_path)
    logging.info("Saved checkpoint to {}".format(model_path))
    # compute training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--vgg', help='Pre-trained vgg model path, download from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--log_step', default=50, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=5000, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logging.info(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logging.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logging.info(config_str)
    logging.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logging.info('Start evaluating...')
        test_dataset = build_dataset(dataset_list=cfg.DATASETS.TEST)
        print("Test dataset size: {}".format(len(test_dataset)))
        do_evaluation(cfg, model, test_dataset, cfg.OUTPUT_DIR)


if __name__ == '__main__':
    main()
