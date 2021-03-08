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

    def plot_interpolation(self, image_path: str, n_steps: int = 7):
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

        interpolation = self.generative_model.interpolate(
            image, baseline, n_steps = n_steps, add_endpoints = True)
        
        labels = {
            0: 'original input',
            1: 'reconstruction',
            len(interpolation) - 2: 'reconstruction',
            len(interpolation) - 1: 'inpainted baseline'
        }

        # Add mask
        interpolation[-1] = PIL.Image.blend(interpolation[-1], opencv_to_pillow_image(mask), 0.3)
        
        plot_image_grid(interpolation, labels)

if __name__ == "__main__":
    latent_ig = Latent_IG()

    latent_ig.plot_interpolation("../data/places_small/Places365_val_00000247.jpg")
    plt.show()