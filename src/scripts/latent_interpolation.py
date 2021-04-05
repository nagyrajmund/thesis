import os
from os.path import join
from typing import List
from PIL.Image import blend
import numpy as np
import cv2

import PIL
from matplotlib import pyplot as plt

from models.wrappers.inpainting import InpaintingModel
from models.wrappers.segmentation import SegmentationModel
from models.wrappers.generation import ImageGenerator
from scripts import utils

def parse_args():
    parser = utils.create_default_argparser(output_dir="outputs/latent_interpolation")

    parser.add_argument("--n_steps", type=int, default=7,
                        help="The number of interpolation steps")
    parser.add_argument("--dilation", type=int, default=10,
                        help="The number of mask dilation iterations")
    args = parser.parse_args()
        
    return args

def main():
    """
    This script plots a linear interpolation between original images and inpainted baselines,
    performed in the latent space of the generative model from DALL-E.
    """
    for file in os.listdir(args.data_dir):
        print(file)
        plot_interpolation(image_path = join(args.data_dir, file))
        
        if args.show_plot:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()
        else:
            plt.savefig(join(args.output_dir, file), bbox_inches='tight')

def plot_interpolation(
    image_path: str, 
    n_steps: int = 7,
    plot_original_imgs: bool = True,
    plot_mask_overlay: bool = True
):
    """
    Plot the latent interpolation betweeen the original image and and its inpainted baseline.

    Args:
        image_path:  The path to the input image
        n_steps:     The number of interpolation steps
    """
    image = cv2.imread(image_path)
    masks = segmentation_model.extract_segmentation(image)

    # TODO(RN): merge masks or use multiple baselines
    mask = masks.max(axis = 0, keepdims = False)

    if args.dilation > 0:
        mask = utils.dilate_mask(mask, n_iters=args.dilation)
        
    baseline = inpainting_model.inpaint(image, mask)

    # Convert OpenCV images to PIL images
    # TODO(RN): a bit quirky 
    image = utils.opencv_to_pillow_image(image)
    # The inpainting network returns the baseline as RGB openCV image,
    # so this is enough to convert it to a PIL image
    baseline = PIL.Image.fromarray(baseline)

    interpolation = generative_model.interpolate(
        image, baseline, n_steps, plot_original_imgs)
    
    if plot_mask_overlay:
        _plot_interpolation_array(interpolation, mask)
    else:
        _plot_interpolation_array(interpolation)

def _plot_interpolation_array(
    img_list: List[PIL.Image.Image], 
    mask: np.ndarray = None
):
    """
    Plot the latent interpolation found in img_list. Additionally, if 'mask'
    is given, its overlay on the original image and the baseline will be shown.
    """
    n_images = len(img_list)

    labels = {
        0: 'original input',
        1: 'reconstruction',
        n_images - 2: 'reconstruction',
        n_images - 1: 'inpainted baseline',
    }

    # If no mask is given, the images will be plotted in one row
    if mask is None:
        return utils.plot_image_grid(img_list, labels)

    # Otherwise, create a two row plot
    n_rows = 2
    n_cols = len(img_list)
    
    _, axes = plt.subplots(n_rows, n_cols, gridspec_kw = {'wspace': 0, 'hspace': 0.1}, figsize=utils.get_figsize(n_rows, n_cols))
    top_axis_row = axes[0]
    bottom_axis_row = axes[1]

    # Turn off all axis ticks
    for _, axis in np.ndenumerate(axes):
        axis.axis('off')

    # Helper function
    def plot_image(axis, image, label):
        axis.imshow(image)
        if label is not None:
            axis.set_title(label)

    # First we plot the images in img_list on the bottom axes
    for i, (axis, image) in enumerate(zip(bottom_axis_row, img_list)):
        label = labels[i] if i in labels else None
        plot_image(axis, image, label)       

    # Then, we plot the overlays for the original and the baseline image on the upper row
    overlay = lambda img, mask : blend(img, utils.opencv_to_pillow_image(mask), alpha = 0.3)
    
    original_with_mask = overlay(img_list[0],  mask)
    baseline_with_mask = overlay(img_list[-1], mask)
    
    plot_image(top_axis_row[0], original_with_mask, "original with mask")
    plot_image(top_axis_row[-1], baseline_with_mask, "baseline with mask")


if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    utils.save_args_to_output_dir(args)    

    inpainting_model = InpaintingModel()
    segmentation_model = SegmentationModel()
    generative_model = ImageGenerator()
    
    main()