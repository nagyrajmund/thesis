from typing import List
from PIL.Image import blend
import numpy as np
import cv2

import PIL
from matplotlib import pyplot as plt

from src.inpainting import InpaintingModel
from src.segmentation import SegmentationModel
from src.interpolation import ImageGenerator
from src.utils import plot_image_grid, opencv_to_pillow_image

class Latent_IG:
    """
    An improved version of integrated gradients that
        1) Creates better baselines by removing objects with an inpainter network; 
        2) Interpolates in the latent space of a generative network.

    Args:
        TODO(RN)
    
    Returns:
        TODO(RN)
    """
    def __init__(self):
        # TODO(RN): create hparams dict to configure all these models
        self.inpainting_model = InpaintingModel()
        self.segmentation_model = SegmentationModel()
        self.generative_model = ImageGenerator()

    def plot_interpolation(self, 
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
        masks = self.segmentation_model.extract_segmentation(image)

        # TODO(RN): merge masks or use multiple baselines
        mask = masks.max(axis = 0, keepdims = False)
        baseline = self.inpainting_model.inpaint(image, mask)

        # Convert OpenCV images to PIL images
        # TODO(RN): a bit quirky 
        image = opencv_to_pillow_image(image)
        # The inpainting network returns the baseline as RGB openCV image,
        # so this is enough to convert it to a PIL image
        baseline = PIL.Image.fromarray(baseline)
        print("----------- interpolating -------------")
        interpolation = self.generative_model.interpolate(
            image, baseline, n_steps, plot_original_imgs)
        print("-----------      DONE     -------------")
        
        if plot_mask_overlay:
            self._plot_interpolation_array(interpolation, mask)
        else:
            self._plot_interpolation_array(interpolation)

    def _plot_interpolation_array(self,
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
            return plot_image_grid(img_list, labels)

        # Otherwise, create a two row plot
        _, axes = plt.subplots(2, len(img_list), gridspec_kw = {'wspace': 0, 'hspace': 0}, sharex='row')
        top_row = axes[0]
        bottom_row = axes[1]

        for _, axis in np.ndenumerate(axes):
            axis.axis('off')
        # Helper function
        def plot_image(axis, image, label):
            axis.imshow(image)

            if label is not None:
                axis.set_title(label)

        # First we plot the images in img_list - this will be the upper row
        for i, (axis, image) in enumerate(zip(axes[0], img_list)):
            label = labels[i] if i in labels else None
            plot_image(axis, image, label)       

        # Then, we plot the overlays for the original and the baseline image in the lower row
        overlay = lambda img, mask : blend(img, opencv_to_pillow_image(mask), 0.3)
        
        original_with_mask = overlay(img_list[0],  mask)
        baseline_with_mask = overlay(img_list[-1], mask)
        
        plot_image(axes[1,0], original_with_mask, "original with mask")
        plot_image(axes[1,-1], baseline_with_mask, "baseline with mask")
        plt.tight_layout()

if __name__ == "__main__":
    latent_ig = Latent_IG()

    latent_ig.plot_interpolation("../data/places_small/Places365_val_00000199.jpg")
    plt.show()