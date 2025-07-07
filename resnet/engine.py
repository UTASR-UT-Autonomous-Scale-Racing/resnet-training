"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import math
import sys

import torch
import utils
from sklearn.metrics import jaccard_score

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(10)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_scores = []
    import cv2

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
        
        for target, prediction, image in zip(targets, predictions, images):
            true_masks = target["masks"].squeeze(1).cpu().numpy()
            pred_masks = (prediction["masks"] > 0.7).squeeze(1).cpu().numpy()

            image = image.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
            overlay = image.copy()
            for pred_mask in pred_masks:
                overlay[pred_mask > 0] = [0, 255, 0]  # Green overlay for predicted mask
            blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            cv2.imshow("Overlay", blended)
            cv2.waitKey(1)
            for true_mask, pred_mask in zip(true_masks, pred_masks):
                iou = jaccard_score(true_mask.flatten(), pred_mask.flatten())
                iou_scores.append(iou)
    cv2.destroyAllWindows()
    mean_iou = sum(iou_scores) / len(iou_scores)
    torch.set_num_threads(n_threads)
    return mean_iou
