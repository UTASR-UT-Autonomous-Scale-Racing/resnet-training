"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import torch

import utils
from engine import train_one_epoch, evaluate
from helpers import get_segmentation_model, MultiObjectMaskDataset
import os
import random

if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # create directory for saving checkpoints if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    train_partition = 0.7
    val_partition = 0.2

    # our dataset has two classes only - background and object
    num_classes = 2
    root = 'data/dataset'
    imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
    masks = list(sorted(os.listdir(os.path.join(root, "targets"))))
    indices = list(range(len(imgs)))
    train_indices = indices[:int(len(indices) * train_partition)]
    val_indices = indices[int(len(indices) * train_partition):int(len(indices) * (train_partition + val_partition))]

    # split the dataset in train and test set
    random.shuffle(train_indices)
    train_imgs = [imgs[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_imgs = [imgs[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]

    # setup preprocessing and reading of images and targets
    dataset_train = MultiObjectMaskDataset(train_transforms=True, imgs=train_imgs, \
                                           image_dir='data/dataset/images', target_dir='data/dataset/targets', masks=train_masks)
    dataset_val = MultiObjectMaskDataset(train_transforms=False, imgs=val_imgs, \
                                         image_dir='data/dataset/images', target_dir='data/dataset/targets', masks=val_masks)


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=4,
        shuffle=False,
        collate_fn=utils.collate_fn
    )


    # get the model using our helper function
    model = get_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        torch.save(model.state_dict(), f"checkpoints/mask_rcnn_model_{epoch}.pth")
        print(f"Model saved for epoch {epoch}")
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        mean_iou = evaluate(model, data_loader_val, device=device)
        print(f"Mean IoU: {mean_iou}")

    torch.save(model.state_dict(), f"checkpoints/mask_rcnn_model.pth")
    print("Model saved")
