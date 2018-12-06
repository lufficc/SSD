import os

import torch
from tqdm import tqdm
from ssd.config import cfg
from ssd.datasets import build_dataset
from ssd.modeling.predictor import Predictor
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.utils.eval_detection_voc import eval_detection_voc

import argparse
import numpy as np


def do_evaluation(cfg, model, test_dataset, output_dir):
    class_names = test_dataset.class_names
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()
    predictor = Predictor(cfg=cfg,
                          model=model,
                          iou_threshold=cfg.TEST.NMS_THRESHOLD,
                          score_threshold=cfg.TEST.CONFIDENCE_THRESHOLD,
                          device=device)

    cpu_device = torch.device("cpu")

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []
    for i in tqdm(range(len(test_dataset))):
        image_id, annotation = test_dataset.get_annotation(i)
        gt_boxes, gt_labels, is_difficult = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)
        gt_difficults.append(is_difficult.astype(np.bool))

        image = test_dataset.get_image(i)
        output = predictor.predict(image)
        boxes, labels, scores = [o.to(cpu_device).numpy() for o in output]

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)

    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficults,
                                iou_thresh=0.5,
                                use_07_metric=True)

    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    print(result_str)
    prediction_path = os.path.join(output_dir, "result.txt")
    with open(prediction_path, "w") as f:
        f.write(result_str)


def evaluation(cfg, weights_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_dataset = build_dataset(dataset_list=cfg.DATASETS.TEST)
    print("Test dataset size: {}".format(len(test_dataset)))

    device = torch.device(cfg.MODEL.DEVICE)
    model = build_ssd_model(cfg, is_test=True)
    model.load(weights_file)
    print('Loaded weights from {}.'.format(weights_file))
    model.to(device)
    do_evaluation(cfg, model, test_dataset, output_dir)


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
