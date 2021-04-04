import os, sys
from contextlib import nullcontext
from typing import List, Dict

import numpy as np
import detectron2
import tensorflow as tf
import cv2

from matplotlib import pyplot as plt
import PIL

def infer_detectron2_class_names(config_file: str) -> List[str]:
    """Return a list of all known class names for the model of the given config file."""
    if config_file.startswith("COCO-InstanceSegmentation"):
        dataset = "coco_2017_train"
    elif config_file.startswith("LVISv0.5-InstanceSegmentation"):
        dataset = "lvis_v0.5_train"
    elif config_file.startswith("LVISv1-InstanceSegmentation"):
        dataset = "lvis_v1_train"
    else:
        raise Exception(f"Could not infer dataset name from config file '{config_file}'")

    return detectron2.data.MetadataCatalog.get(dataset).thing_classes


class LogDisabler:
    """Context manager to disable prints and tensorflow logs e.g. in a function call.""" 
    def __enter__(self):
        tf.get_logger().setLevel('ERROR')
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.get_logger().setLevel('DEBUG')
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_log_manager(disable_logs: bool):
    """
    Return a context manager for hiding print() outputs and TF logs if 'disable_prints' is True,
    otherwise return a dummy context manager that does nothing. 
    """
    if disable_logs:
        return LogDisabler()
    else:
        return nullcontext()

def plot_image_grid(images: np.ndarray, labels: Dict[int, str] = {}):
    """
    Plot a 1D grid of images and use the 'labels' as their titles (when given).

    Args:
        images: an array of multiple RGB images with shape (N, H, W, C)
        labels: a dictionary of image indices and the corresponding labels
    """
    n = len(images)
    
    # if there's only one image then we can't use plt.subplots() -> just plot it normally
    if n == 1:
        plt.imshow(images[0])
        
        if 0 in labels:
            plt.title(labels[0])
        
        return

    _, axes = plt.subplots(1, n, gridspec_kw = {'wspace': 0, 'hspace': 0})

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
        
        if i in labels:
            ax.set_title(labels[i])

def plot_image_grid(images: np.ndarray, labels: Dict[int, str] = {}):
    """
    Plot a 1D grid of images and use the 'labels' as their titles (when given).

    Args:
        images: an array of multiple RGB images with shape (N, H, W, C)
        labels: a dictionary of image indices and the corresponding labels
    """
    n = len(images)
    
    # if there's only one image then we can't use plt.subplots() -> just plot it normally
    if n == 1:
        plt.imshow(images[0])
        
        if 0 in labels:
            plt.title(labels[0])
        
        return

    _, axes = plt.subplots(1, n, gridspec_kw = {'wspace': 0, 'hspace': 0})

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
        
        if i in labels:
            ax.set_title(labels[i])

def opencv_to_pillow_image(opencv_image: np.ndarray) -> PIL.Image.Image:
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pillow_image = PIL.Image.fromarray(opencv_image)

    return pillow_image
