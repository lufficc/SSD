import json
import logging
import os


def coco_evaluation(dataset, predictions, output_dir):
    coco_results = []
    for i, (boxes, labels, scores) in enumerate(predictions):
        image_id, annotation = dataset.get_annotation(i)
        class_mapper = dataset.contiguous_id_to_coco_id
        if labels.shape[0] == 0:
            continue

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": class_mapper[labels[k]],
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    iou_type = 'bbox'
    json_result_file = os.path.join(output_dir, iou_type + ".json")
    logger = logging.getLogger("SSD.inference")
    logger.info('Writing results to {}...'.format(json_result_file))
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    from pycocotools.cocoeval import COCOeval
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
