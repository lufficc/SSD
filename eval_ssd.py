import os

import torch
from tqdm import tqdm
from ssd.config import cfg
from ssd.datasets import build_dataset
from ssd.datasets.evaluation import evaluate
from ssd.modeling.predictor import Predictor
from ssd.modeling.vgg_ssd import build_ssd_model

import argparse


def do_evaluation(cfg, model, output_dir):
    test_datasets = build_dataset(dataset_list=cfg.DATASETS.TEST, is_test=True)
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()
    predictor = Predictor(cfg=cfg,
                          model=model,
                          iou_threshold=cfg.TEST.NMS_THRESHOLD,
                          score_threshold=cfg.TEST.CONFIDENCE_THRESHOLD,
                          device=device)

    cpu_device = torch.device("cpu")
    for dataset_name, test_dataset in zip(cfg.DATASETS.TEST, test_datasets):
        print("Test dataset {} size: {}".format(dataset_name, len(test_dataset)))
        predictions = []
        for i in tqdm(range(len(test_dataset))):
            image = test_dataset.get_image(i)
            output = predictor.predict(image)
            boxes, labels, scores = [o.to(cpu_device).numpy() for o in output]
            predictions.append((boxes, labels, scores))
        final_output_dir = os.path.join(output_dir, dataset_name)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))
        evaluate(dataset=test_dataset, predictions=predictions, output_dir=final_output_dir)


def evaluation(cfg, weights_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device(cfg.MODEL.DEVICE)
    model = build_ssd_model(cfg, is_test=True)
    model.load(weights_file)
    print('Loaded weights from {}.'.format(weights_file))
    model.to(device)
    do_evaluation(cfg, model, output_dir)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC Dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--weights", type=str, help="Trained weights.")
    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
    evaluation(cfg, weights_file=args.weights, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
