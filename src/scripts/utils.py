from argparse import ArgumentError, ArgumentParser, Namespace
import os, sys
from contextlib import nullcontext
from typing import List, Dict, Union
import shutil
import numpy as np
import detectron2
import tensorflow as tf
import cv2
import json
from os.path import join
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
import PIL
from typing import Tuple
from pprint import pprint
from matplotlib.axes import Axes

def create_default_argparser(**override_defaults_kwargs) -> ArgumentParser:
    """
    Return an ArgumentParser with common arguments such as the input/output folders.

    If 'override_defaults_kwargs' is provided, it will be used to change the
    default values of the arguments.
    """
    parser = ArgumentParser(add_help=True)
    
    parser.add_argument("--data_dir", type=str, default="../../data/places_small",
                        help="The folder where the input images will be read from")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="The folder where the script's outputs will be saved")

    parser.add_argument("--show_plot", action="store_true",
                        help="If set,the script's results are shown on the screen in addition to saving them to 'output_dir'")

    parser.set_defaults(**override_defaults_kwargs)

    return parser

def save_args_to_output_dir(args: Namespace, print_args: bool = True):
    """
    Save the given command-line arguments to their 'output_dir' folder.
    """
    if print_args:
        pprint(args.__dict__, indent=4, compact=False)

    with open(join(args.output_dir, "cmd_args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=2)

def create_output_dir(args: Namespace):
    """
    Create the 'output_dir' folder. If it already exists, it is emptied upon the user's approval.
    """
    # If show_plot is True, no results will be saved
    if args.show_plot:
        return
    
    if args.output_dir is None:
        raise ArgumentError(args.output_dir, "Please set the result folder with the --output_dir option!")

    if os.path.exists(args.output_dir):
        cmd = input(f"\nWARNING: output dir '{args.output_dir}' already exists!" + \
                     "\nType 'ok' to delete it and anything else to quit: ")
        if cmd != 'ok':
            exit()
        else:
            print()
            shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir)

def infer_detectron2_class_names(config_file: str) -> List[str]:
    """
    Return a list of all known class names for the model of the given config file.
    """
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
    """
    Context manager to disable prints and tensorflow logs e.g. in a function call.
    """ 
    def __enter__(self):
        tf.get_logger().setLevel('ERROR')
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.get_logger().setLevel('DEBUG')
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_log_manager(disable_logs: bool = True) -> Union[LogDisabler, nullcontext]:
    """
    Return a context manager for hiding print() outputs and TF logs if 'disable_prints' is True,
    otherwise return a dummy context manager that does nothing. 
    """
    if disable_logs:
        return LogDisabler()
    else:
        return nullcontext()

def plot_image(axis: Axes, image: PIL.Image.Image, title: str = None, xlabel: str = None):
    """
    Plot the image on the axis with the given title.
    """
    axis.imshow(image)

    if title is not None:
        axis.set_title(title)
    
    if xlabel is not None:
        axis.set_xlabel(xlabel)

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
    """
    Convert the given openCV image to a Pillow image (from a BGR to an RGB channel order).
    """
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pillow_image = PIL.Image.fromarray(opencv_image)

    return pillow_image

def get_figsize(n_rows: int, n_cols: int, subplot_size: int = 5) -> Tuple[float, float]:
    """
    Return an appropriate figure size for a plot with the given number of rows and columns.
    """
    return (n_cols * subplot_size, n_rows * subplot_size)
    
def dilate_mask(mask: np.ndarray, n_iters: int, kernel_size: int = 3) -> np.ndarray:
    """
    Pad the given mask using the dilation operation.

    Args:
        mask:        A binary mask
        n_iters:     The number of dilation iterations
        kernel_size: The size of the kernel matrix for the dilation
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    return cv2.dilate(mask, kernel, iterations = n_iters)