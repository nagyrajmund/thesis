"""
This script filters the Places365 validation dataset to select a fixed number of images
for the experiments in the thesis.

We try to segment the objects on each image using the 4 instance segmentation networks
described in 
If the size of the combined segmentation maps is less than [10%] of the input size,
the image is discarded.
"""
import tensorflow as tf
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
n_images                = 3000  # The number of desired images
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
            f"\nWARNING: output dir '{segmentation_output_dir}' already exists!\n" + \
            f"Press 'd' to delete and 's' to skip the segmentation, anything else will quit."
        )
    
        if cmd.lower() == "d":
            shutil.rmtree(segmentation_output_dir)
        elif cmd.lower() == "s":
            return
        else:
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
    if os.path.exists(inpainting_output_dir):
        cmd = input(
            f"\nWARNING: output dir '{inpainting_output_dir}' already exists!\n" + \
            f"Press 'd' to delete and 's' to skip the inpainting, anything else will quit."
        )
    
        if cmd.lower() == "d":
            shutil.rmtree(inpainting_output_dir)
        elif cmd.lower() == "s":
            return
        else:
            exit()
            
    os.makedirs(inpainting_output_dir)

    # Only consider images that have segmentations
    filenames = os.listdir(segmentation_output_dir)
    progress_bar = tqdm(filenames, desc="Processing images")
    inpainting_model = InpaintingModel()
    
    with inpainting_model.log_manager:
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        temp_image = cv2.imread(join(image_input_dir, filenames[0]))    
        h, w, _ = temp_image.shape
        grid = 8
        
        # Preprocess the current image
        transform = lambda img : np.expand_dims(img[:h//grid*grid, :w//grid*grid, :], 0)
        with tf.Session(config=sess_config) as sess:
            # Dirty, dirty tensorflow stuff to load the model 
            sess.run(initialize_model(inpainting_model, transform(temp_image)))
            
            for filename in progress_bar:
                # Open the input and it segmentation
                original_image = cv2.imread(join(image_input_dir, filename))
                segmentation = cv2.imread(join(segmentation_output_dir, filename))
                
                # Preprocess and concatenate them
                image = transform(original_image)
                mask = transform(segmentation)
                input_image = np.concatenate([image, mask], axis=2)        
                input_image = tf.constant(input_image, dtype=tf.float32)
                
                # Build the computational graph
                output = inpainting_model.model.build_server_graph(
                    inpainting_model.flags, input_image, reuse=tf.AUTO_REUSE)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                
                # Run the computational graph
                inpainted_image = sess.run(output)[0]

                # Convert back to BGR and save with openCV
                inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(join(inpainting_output_dir, filename), inpainted_image)

        inpainted_image = inpainting_model.inpaint(input_image, segmentation)


def initialize_model(model, image):
    "To be used as 'sess.run(initialize_model(...))'."
    input_image = tf.constant(image, dtype=tf.float32)
    output = model.model.build_server_graph(
            model.flags, input_image, reuse=tf.AUTO_REUSE)
    
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable("../../utils/deepfill_checkpoint/", from_name)
        assign_ops.append(tf.assign(var, var_value))

    return assign_ops

if __name__ == "__main__":
    create_segmentation_dataset()
    create_inpainted_dataset()