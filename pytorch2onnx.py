import os
import glob
import time
import argparse
import onnxruntime  # install onnnxruntime from pip

import torch
import numpy as np
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.data.transforms import build_transforms
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.modeling.detector import build_detection_model
from ssd.modeling.box_head.inference import PostProcessorOnnx


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, onnx_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)
    device = "cpu" if not torch.cuda.is_available() else device

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    # get model ready for onnx export
    mkdir(onnx_dir)
    model_onnx = build_detection_model(cfg)
    model_onnx = model_onnx.to(device)
    checkpointer_onnx = CheckPointer(model_onnx, save_dir=cfg.OUTPUT_DIR)
    checkpointer_onnx.load(ckpt, use_latest=ckpt is None)
    # replace the SSD box head postprocessor with the onnx version for exporting
    model_onnx.box_head.post_processor = PostProcessorOnnx(cfg)
    model_onnx.eval()

    # export with ONNX
    # onnx modle takes the name of the pth ckpt file
    model_onnx_name = os.path.basename(ckpt).split('.')[0] + ".onnx"
    model_onnx_path = os.path.join(onnx_dir, model_onnx_name)
    if not os.path.exists(model_onnx_path):
        print(f'Model exported as onnx to {model_onnx_path}')
        dummy_input = torch.zeros(
            [1, 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE]).to(device)
        torch.onnx.export(model_onnx,
                          dummy_input,
                          model_onnx_path,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=11,
                          input_names=['input'],
                          output_names=['boxes', 'scores', 'labels'],
                          dynamic_axes={
                              'input': {0: 'batch_size', 2: "height", 3: "width"}},
                          verbose=False)

    # load exported onnx model for inference test
    print(
        f'Loading exported onnx model from {model_onnx_path} for inference comparison test')
    onnx_runtime_sess = onnxruntime.InferenceSession(model_onnx_path)

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)

        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start
        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('Pytorch: ({:04d}/{:04d}) {}: {}'.format(i + 1,
                                                       len(image_paths), image_name, meters))
        drawn_image = draw_boxes(image, boxes, labels,
                                 scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(
            os.path.join(output_dir, "pytorch_" + image_name))

        """
        Compute ONNX Runtime output prediction
        """

        start = time.time()
        ort_inputs = {onnx_runtime_sess.get_inputs()[0].name: np.array(images)}
        boxes, scores, labels = onnx_runtime_sess.run(None, ort_inputs)
        inference_time = time.time() - start

        indices = scores > score_threshold
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
        # resize bounding boxes to size of the original image
        boxes[:, 0::2] *= (width)
        boxes[:, 1::2] *= (height)
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('Onnx:    ({:04d}/{:04d}) {}: {}'.format(i + 1,
                                                       len(image_paths),
                                                       image_name, meters))
        drawn_image = draw_boxes(image, boxes, labels,
                                 scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(
            os.path.join(output_dir, "onnx_" + image_name))


def main():
    parser = argparse.ArgumentParser(
        description="SSD pytorch to onnx conversion")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo',
                        type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result',
                        type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--onnx_dir", default='models',
                        type=str, help='Specify a image dir to save exported onnx model.')
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')

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

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             onnx_dir=args.onnx_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
