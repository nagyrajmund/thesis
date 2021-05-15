from shutil import rmtree
from os.path import join
import os

from src.models.wrappers.segmentation import InstanceSegmentationModel
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

"""
This script compares the segmentation quality of the three
available instance segmentaiton models from Detectron2.
"""

# Create output_dir
output_dir = "./compare_segmentation_models"
if os.path.exists(output_dir):
    rmtree(output_dir)
os.makedirs(output_dir)
print(f"Results will be saved to {os.path.abspath(output_dir)}")

model_configs = [
    "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
    "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
]   

print("Initializing models...")
models = [InstanceSegmentationModel(config) for config in model_configs]

image_dir = "../../data/places365/validation_medium"
images = sorted(os.listdir(image_dir))
n_images = len(images)
# Number of images to show at the same time
batch_size = 6
n_models = len(model_configs)

for batch_start in tqdm(range(0, n_images, batch_size), desc=f"Processing {batch_size} images at a time"):
    _, axes = plt.subplots(n_models, batch_size)
    
    for model_idx in tqdm(range(n_models), leave = False, desc=f"Generating segmentation with {n_models} different models"):
        for image_idx in tqdm(range(batch_size)):
            image = cv2.imread(join(image_dir, images[batch_start + image_idx]))
            overlay = models[model_idx].draw_segmentation_overlay(image)
            
            curr_axis = axes[model_idx, image_idx]
            
            # # Set title at first image
            # if image_idx == 0:
            #     curr_axis.set_title(
            #         os.path.split(model_configs[model_idx])[1], 
            #         fontdict = {'fontsize': 12}, 
            #         loc = 'left',
            #         pad = '2'
            #     )

            curr_axis.imshow(overlay)
            curr_axis.axis('off')

    plt.tight_layout()
    plt.savefig(join(output_dir, f"{images[batch_start]}"), bbox_inches="tight", transparent=True, dpi=600)
    print(f"Saved comparison to {join(output_dir, f'{images[batch_start]}_{batch_size}')}")