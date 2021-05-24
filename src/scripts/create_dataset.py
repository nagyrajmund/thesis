"""
This script filters the Places365 validation dataset to select a fixed number of images
for the experiments in the thesis.

We try to segment the objects on each image using the 4 instance segmentation networks
described in 
If the size of the combined segmentation maps is less than [10%] of the input size,
the image is discarded.
"""
from scripts.integrated_gradients import update_progress_bar
import shutil
from src.models.wrappers.inpainting import InpaintingModel
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
inpainting_output_dir   = "../../data/thesis/places365/inpaintings"
dilation                = 7
# --------------------------------------------------------------------------

def create_segmentation_dataset():

    if os.path.exists(segmentation_output_dir):
        cmd = input(
            f"\nWARNING: output dir '{segmentation_output_dir}' already exists!" + \
"""
Type 'delete' to delete the folder,
    'skip' to skip the segmentation, and
    'keep' to continue saving to the folder.
"""
        )
    
        if cmd == "delete":
            shutil.rmtree(segmentation_output_dir)
        elif cmd == "skip":
            return
        elif cmd != "keep":
            exit()

    os.makedirs(segmentation_output_dir)

    print("Initializing segmentation models...")
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

def create_inpainted_dataset():
    if not os.path.exists(inpainting_output_dir):
        os.makedirs(inpainting_output_dir)
    else:
        cmd = input(
            f"\nWARNING: output dir '{inpainting_output_dir}' already exists!" + \
"""
Type 'delete' to delete the folder,
    'skip' to skip the inpainting, and
    'keep' to continue saving to the folder.
"""
        )
    
        if cmd == "delete":
            shutil.rmtree(inpainting_output_dir)
        elif cmd == "skip":
            return
        elif cmd != "keep":
            exit()

    # Only consider images that have segmentations
    filenames = os.listdir(segmentation_output_dir)
    progress_bar = tqdm(filenames, desc="Processing images")
    inpainting_model = InpaintingModel()
    for filename in progress_bar:
        input_image = cv2.imread(join(image_input_dir, filename))
        segmentation = cv2.imread(join(segmentation_output_dir, filename))

        inpainted_image = inpainting_model.inpaint(input_image, segmentation)
        # Convert back to BGR and save
        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(join(inpainting_output_dir, filename), inpainted_image)

if __name__ == "__main__":
    create_segmentation_dataset()
    create_inpainted_dataset()