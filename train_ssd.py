import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ssd.config import cfg
from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.engine.inference import do_evaluation
from ssd.engine.trainer import do_train
from ssd.modeling.data_preprocessing import TrainAugmentation
from ssd.modeling.multibox_loss import MultiBoxLoss
from ssd.modeling.ssd import MatchPrior
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.module.prior_box import PriorBox
from ssd.utils import distributed_util
from ssd.utils.logger import setup_logger
from ssd.utils.lr_scheduler import WarmupMultiStepLR
from ssd.utils.misc import str2bool


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = build_ssd_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.resume:
        logger.info("Resume from the model {}".format(args.resume))
        model.load(args.resume)
    else:
        logger.info("Init from base net {}".format(args.vgg))
        model.init_from_base_net(args.vgg)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # -----------------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------------
    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    # -----------------------------------------------------------------------------
    # Criterion
    # -----------------------------------------------------------------------------
    criterion = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)

    # -----------------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------------
    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=milestones,
                                  gamma=cfg.SOLVER.GAMMA,
                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS)

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    train_transform = TrainAugmentation(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN)
    target_transform = MatchPrior(PriorBox(cfg)(), cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE, cfg.MODEL.THRESHOLD)
    train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform, target_transform=target_transform)
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=False)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER // args.num_gpus)
    train_loader = DataLoader(train_dataset, num_workers=4, batch_sampler=batch_sampler)

    return do_train(cfg, model, train_loader, optimizer, scheduler, criterion, device, args)


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
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
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger = setup_logger("SSD", distributed_util.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, cfg.OUTPUT_DIR, distributed=args.distributed)


if __name__ == '__main__':
    main()
