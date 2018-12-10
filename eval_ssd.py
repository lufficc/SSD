import argparse
import logging
import os

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.utils import distributed_util
from ssd.utils.logger import setup_logger


def evaluation(cfg, weights_file, output_dir, distributed):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_ssd_model(cfg, is_test=True)
    model.load(weights_file)
    logger = logging.getLogger("SSD.inference")
    logger.info('Loaded weights from {}.'.format(weights_file))
    model.to(device)
    do_evaluation(cfg, model, output_dir, distributed)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--weights", type=str, help="Trained weights.")
    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("SSD", distributed_util.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, weights_file=args.weights, output_dir=args.output_dir, distributed=distributed)


if __name__ == '__main__':
    main()
