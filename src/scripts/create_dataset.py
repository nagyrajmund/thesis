"""
This script filters the Places365 validation dataset to select a fixed number of images
for the experiments in the thesis.

We try to segment the objects on each image using the 4 instance segmentation networks
described in 
If the size of the combined segmentation maps is less than [10%] of the input size,
the image is discarded.
"""
from src.scripts import utils
from shutil import rmtree
import os
import torch
from os.path import join
import cv2
from models.wrappers.segmentation import InstanceSegmentationModel
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

segmentation_net_configs = [
    "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
    "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
]

device                  = "cuda" if torch.cuda.is_available() else "cpu"
n_images                = 100  # The number of desired images
starting_idx            = 0    # The first image to start with
confidence_threshold    = 0.3  # The minimum confidence for accepting an object segmentation
image_input_dir         = "../../data/places365/validation_full"
segmentation_output_dir = "../../data/thesis/places365/segmentations"
dilation                = 7
# --------------------------------------------------------------------------
if os.path.exists(segmentation_output_dir):
    rmtree(segmentation_output_dir)
os.makedirs(segmentation_output_dir)

print("Initializing models...")
segmentation_models = [
    InstanceSegmentationModel(config, device, confidence_threshold)
    for config in segmentation_net_configs
]

filenames = sorted(os.listdir(image_input_dir))[starting_idx:]

n_saved_images = 0
progress_bar = tqdm(filenames, desc="Processing images")
for filename in progress_bar:
    input_image = cv2.imread(join(image_input_dir, filename))

    print("Extracting segmentations...")
    segmentation_masks = [model.extract_segmentation(input_image)
                          for model in segmentation_models]
    
    # Filter out empty masks
    segmentation_masks = [mask for mask in segmentation_masks if mask is not None]
    
    # Skip image if no objects were found
    if len(segmentation_masks) == 0:
        continue
   
    # Merge the segmentation masks of the different models
    segmentation_masks = np.concatenate(segmentation_masks, axis=0)
    segmentation_masks = np.max(segmentation_masks, axis=0)
    # Dilate the results
    segmentation_masks = utils.dilate_mask(segmentation_masks, n_iters = 7)

    
    # Discard the segmentation maps that are too small
    image_size = input_image.shape[0] * input_image.shape[1]
    if segmentation_masks.sum() < 0.1 * image_size:
        continue
    
    cv2.imwrite(join(segmentation_output_dir, filename), segmentation_masks * 255)
    
    n_saved_images += 1
    if n_saved_images == n_images:
        print(f"Saved all images! Last file: {filename}")
        break