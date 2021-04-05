import os
from os.path import join
from typing import List
from PIL.Image import blend
import numpy as np
import cv2

import PIL
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from models.wrappers.inpainting import InpaintingModel
from models.wrappers.segmentation import InstanceSegmentationModel
from models.wrappers.generation import ImageGenerator
from models.wrappers.classification import SceneRecognitionModel
from scripts import utils
from tqdm import tqdm

def parse_args():
    parser = utils.create_default_argparser(output_dir="outputs/latent_interpolation")

    parser.add_argument("--show_classifier_outputs", action="store_true",
                        help="If set, an additional row will be shown with the classifier's outputs")
    parser.add_argument("--n_steps", type=int, default=7,
                        help="The number of interpolation steps")
    parser.add_argument("--dilation", type=int, default=7,
                        help="The number of mask dilation iterations")
    args = parser.parse_args()
        
    return args

def main():
    """
    This script plots a linear interpolation between original images and inpainted baselines,
    performed in the latent space of the generative model from DALL-E.
    """
    progress_bar = tqdm(os.listdir(args.data_dir))
    for file in progress_bar:
        progress_bar.set_description(file)

        image_file = join(args.data_dir, file)

        input_space_axes, latent_space_axes, classifier_axes = create_plot()

        plot_interpolation(
            image_file, 
            interpolation_axes=latent_space_axes,
            original_img_axes=input_space_axes,
            model_output_axes=classifier_axes
        )
        
        if args.show_plot:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show(block=False)
        else:
            plt.savefig(join(args.output_dir, file), bbox_inches='tight')

def create_plot():
    # There can be three rows: 1) input space 2) latent space 3) model outputs
    n_rows = 3 if args.show_classifier_outputs else 2
    n_cols = args.n_steps

    _, axes = plt.subplots(
        n_rows, n_cols, 
        gridspec_kw = {'wspace': 0.025, 'hspace': 0.1}, 
        figsize=utils.get_figsize(n_rows, n_cols)
    )
    
    input_space_axes = axes[0]
    latent_space_axes = axes[1]
    classifier_axes = axes[2] if n_rows == 3 else None

    # Turn off all axis ticks
    for _, axis in np.ndenumerate(axes):
        axis.axis('off')
    
    return input_space_axes, latent_space_axes, classifier_axes

def plot_interpolation(
    image_file: str,
    interpolation_axes: Axes,
    original_img_axes: Axes,
    model_output_axes: Axes
):
    """
    Plot the latent interpolation betweeen the original image and and its inpainted baseline.

    Args:
        image_file:  The path to the input image
        n_steps:     The number of interpolation steps
    """
    original = cv2.imread(image_file)
    masks = segmentation_model.extract_segmentation(original)

    # TODO(RN): merge masks or use multiple baselines
    mask = masks.max(axis = 0, keepdims = False)

    if args.dilation > 0:
        mask = utils.dilate_mask(mask, n_iters=args.dilation)
        
    baseline = inpainting_model.inpaint(original, mask)

    # Convert OpenCV images to PIL images
    # TODO(RN): a bit quirky 
    original_image = utils.opencv_to_pillow_image(original)
    # The inpainting network returns the baseline as RGB openCV image,
    # so this is enough to convert it to a PIL image
    baseline_image = PIL.Image.fromarray(baseline)

    interpolation_imgs = generative_model.interpolate(
        original_image, baseline_image, args.n_steps, add_original_images=False)

    _plot_original_and_baseline_img(original_image, baseline_image, mask, original_img_axes)
    _plot_interpolation_array(interpolation_imgs, interpolation_axes)

def _plot_original_and_baseline_img(
    original_image: PIL.Image.Image,
    baseline_image: PIL.Image.Image,
    mask: np.ndarray,
    original_img_axes: Axes
):
    """
    TODO(RN) update documentation
    """
     # Then, we plot the overlays for the original and the baseline image on the upper row
    overlay = lambda img, mask : blend(img, utils.opencv_to_pillow_image(mask), alpha = 0.3)
    
    original_with_mask = overlay(original_image,  mask)
    baseline_with_mask = overlay(baseline_image, mask)
    
    utils.plot_image(original_img_axes[0], original_image, "original image")
    utils.plot_image(original_img_axes[1], original_with_mask, "original image with mask")
    utils.plot_image(original_img_axes[-2], baseline_with_mask, "baseline image with mask")
    utils.plot_image(original_img_axes[-1], baseline_image, "baseline image")


def _plot_interpolation_array(
    interpolation_imgs: List[PIL.Image.Image],
    interpolation_axes: Axes
):
    """
    TODO(RN) update documentation

    Plot the latent interpolation found in img_list. Additionally, if 'mask'
    is given, its overlay on the original image and the baseline will be shown.
    """
    # Plot the latent interpolation between the reconstructions of the 
    # input and the baseline images
    for i, (axis, image) in enumerate(zip(interpolation_axes, interpolation_imgs)):
        title = "reconstruction" if i == 0 or i == args.n_steps - 1 else None
        utils.plot_image(axis, image, title)


if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    utils.save_args_to_output_dir(args)    

    inpainting_model = InpaintingModel()
    segmentation_model = InstanceSegmentationModel()
    generative_model = ImageGenerator()
    classifier = SceneRecognitionModel()

    main()