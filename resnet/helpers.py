"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import transforms as my_transforms
from itertools import permutations


base_transforms = [
    [],  # Includes no transforms for inference and training
    [my_transforms.RandomHorizontalFlip(0.5)],
    [my_transforms.RandomIoUCrop(0.5)],
    [my_transforms.RandomPhotometricDistort(0.5)],
    [my_transforms.RandomShortestSize(600, 800)],
    [my_transforms.RandomZoomOut(0.5)],
    [my_transforms.RandomShortestSize(600, 800)],

]

# Generate all permutations of the base transforms
available_transforms = base_transforms #+ [list(p) for p in permutations(base_transforms) if p]


class MultiObjectMaskDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, image_dir, target_dir=None, masks=None, inference=False, train_transforms=False):
        self.inference = inference
        assert inference is False and target_dir is not None and masks is not \
            None or inference is True, "Training mode requires target_dir and masks"
        # imgs, masks must be aligned
        self.img_dir = image_dir
        self.target_dir = target_dir
        self.imgs = imgs
        self.masks = masks
        self._original_len = len(self.imgs)
        if train_transforms:
            self.transforms = [get_transform(train_transforms, i) for i in range(len(available_transforms))]
            self.len = self._original_len * (len(self.transforms))
        else:
            self.transforms = [get_transform(train_transforms)]
            self.len = self._original_len

    def __getitem__(self, idx):
        transform_type = idx // self._original_len
        img_idx = idx % self._original_len

        # load images and masks from disk
        img_path = os.path.join(self.img_dir, self.imgs[img_idx])
        img = read_image(img_path)
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        target = {}
        if not self.inference:
            mask_path = os.path.join(self.target_dir, self.masks[img_idx])
            mask = read_image(mask_path)[0]  # Use only the first channel (grayscale)
            # instances are encoded as different colors/grey values
            obj_ids = torch.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            num_objs = len(obj_ids)

            # split the color-encoded mask into a set
            # of binary masks
            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = idx
            # CocoEvaluator expects the following:
            # get bounding box coordinates for each mask
            boxes = masks_to_boxes(masks)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            # build target dictionary
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            target["boxes"][area <= 0, 2:] += 1  # Adjust boxes if area is not positive
            target["masks"] = tv_tensors.Mask(masks)
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

        if self.inference:
            img = self.transforms[transform_type](img)
            
            return img, {}
        else:
            img, target = self.transforms[transform_type](img, target)

            return img, target

    def __len__(self):
        return self.len

def get_segmentation_model(num_classes):
    # load an instance classification model
    backbone = torchvision.models.resnet18(weights="DEFAULT")
    # FasterRCNN needs to know the number of output channels in a backbone.
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    backbone.out_channels = 512
    # generate anchor boxes for the image 
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    # ROI Pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    # build the Faster R-CNN model
    model = torchvision.models.detection.MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256 # little to no effect on performance
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train, transform_type=0):
    transforms = []
    if train:
        transforms.extend(available_transforms[transform_type])
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
