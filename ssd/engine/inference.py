import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm
from ssd.data.datasets import build_dataset
from ssd.data.datasets.evaluation import evaluate
from ssd.modeling.predictor import Predictor
from ssd.modeling.ssd import SSD

from ssd.utils import distributed_util


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = distributed_util.all_gather(predictions_per_gpu)
    if not distributed_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def _evaluation(cfg, dataset_name, test_dataset, predictor, distributed, output_dir):
    """ Perform evaluating on one dataset
    Args:
        cfg:
        dataset_name: dataset's name
        test_dataset: Dataset object
        predictor: Predictor object, used to to prediction.
        distributed: whether distributed evaluating or not
        output_dir: path to save prediction results
    Returns:
        evaluate result
    """
    cpu_device = torch.device("cpu")
    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(test_dataset)))
    indices = list(range(len(test_dataset)))
    if distributed:
        indices = indices[distributed_util.get_rank()::distributed_util.get_world_size()]

    # show progress bar only on main process.
    progress_bar = tqdm if distributed_util.is_main_process() else iter
    logger.info('Progress on {} 0:'.format(cfg.MODEL.DEVICE.upper()))
    predictions = {}
    for i in progress_bar(indices):
        image = test_dataset.get_image(i)
        output = predictor.predict(image)
        boxes, labels, scores = [o.to(cpu_device).numpy() for o in output]
        predictions[i] = (boxes, labels, scores)
    distributed_util.synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not distributed_util.is_main_process():
        return

    final_output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))
    return evaluate(dataset=test_dataset, predictions=predictions, output_dir=final_output_dir)


def do_evaluation(cfg, model, output_dir, distributed):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    assert isinstance(model, SSD), 'Wrong module.'
    test_datasets = build_dataset(dataset_list=cfg.DATASETS.TEST, is_test=True)
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()
    predictor = Predictor(cfg=cfg,
                          model=model,
                          iou_threshold=cfg.TEST.NMS_THRESHOLD,
                          score_threshold=cfg.TEST.CONFIDENCE_THRESHOLD,
                          device=device)
    # evaluate all test datasets.
    logger = logging.getLogger("SSD.inference")
    logger.info('Will evaluate {} dataset(s):'.format(len(test_datasets)))
    metrics = {}
    for dataset_name, test_dataset in zip(cfg.DATASETS.TEST, test_datasets):
        metric = _evaluation(cfg, dataset_name, test_dataset, predictor, distributed, output_dir)
        metrics[dataset_name] = metric
        distributed_util.synchronize()
    return metrics
