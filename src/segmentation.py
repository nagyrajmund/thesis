import numpy as np
import cv2
import torch
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
setup_logger()

import PIL
from skimage.util import compare_images
import matplotlib.pyplot as plt
from typing import Tuple

from inpainting import InpaintingModel
from src.utils import infer_detectron2_class_names
import matplotlib as mpl

def binary_masks_to_opencv_images(masks):
    masks = masks.numpy()
    masks = [ PIL.Image.fromarray(mask).convert('RGB') for mask in masks ]
    masks = np.stack(masks)
    
    # Convert to BGR
    masks = masks[:, :, :, ::-1]

    return masks

class SegmentationModel:
    """
    A wrapper around pretrained instance segmentation models from the Detectron2 model zoo.

    Args:
        model_config:   The name of the model's config file in the model zoo
        device:         The device of the model ('cpu' or e.g. 'cuda')
        threshold:      The minimum threshold for accepting class predictions as positive
    
    Returns:
        TODO(RN)
    """
    def __init__(self,
        config_file : str   = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        device      : str   = "cpu",
        threshold   : float = 0.3
    ):
        self.threshold = threshold
        self.class_names = infer_detectron2_class_names(config_file)
        
        self.config = detectron2.config.get_cfg()
        self.config.merge_from_file(model_zoo.get_config_file(config_file))
        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        self.config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.config.MODEL.DEVICE = device
        
        self.predictor = DefaultPredictor(self.config)

    def extract_segmentation(self,
        image: np.ndarray,
        return_labels: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image:          An OpenCV image of shape (H, W, C)
            return_labels:  If True, also return the class labels for each mask
        
        Returns:
            pred_masks:     the binary instance segmentation masks of shape (N, H, W)
            pred_classes (if return_labels is True):    the corresponding class labels of shape (N)
        """
        outputs = self.predictor(image)["instances"]
    
        # The instance masks of shape (N, H, W) containing bool values
        pred_masks = outputs.pred_masks
        pred_masks = binary_masks_to_opencv_images(pred_masks)
        # The vector of N class labels
        pred_classes = outputs.pred_classes
        
        if return_labels:
            return pred_masks, pred_classes
        else:
            return pred_masks

    def draw_segmentation_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        Draw a segmentation map on the given image along with the 
        class labels and probabilites.

        Args:
            image:  An OpenCV image of shape (H, W, C)
        
        Returns:
            A matplotlib image of the same shape, showing the segmentation results.
        """
        outputs = self.predictor(image)["instances"]
        metadata = MetadataCatalog.get(self.config.DATASETS.TRAIN[0])
        visualizer = Visualizer(image, metadata, scale=1.2)

        overlay = visualizer.draw_instance_predictions(outputs.to("cpu"))
        # Convert from BGR to RGB
        overlay = overlay.get_image()[:, :, ::-1]
        
        return overlay

#TODO(RN) remove
if __name__ == "__main__":
    inpainting_model = InpaintingModel()
    segmentation_model = SegmentationModel(threshold=0.5)
    image = cv2.imread("../data/places_small/Places365_val_00000173.jpg")
    
    overlay = segmentation_model.draw_segmentation_overlay(image)
    plt.imshow(overlay)
    plt.show()

    masks, labels = segmentation_model.extract_segmentation(image, return_labels=True)
    
    for mask, label in zip(masks, labels):
        # Create RGB Pillow Image from binary mask

        result = inpainting_model.inpaint(image, mask)

        ax = plt.subplot(131)
        ax.imshow(image[:,:,::-1])
        ax.imshow(mask, alpha = 0.2)
        plt.title("original image and mask ({})".format(
            segmentation_model.class_names[label])
        )

        ax = plt.subplot(132)
        ax.imshow(result)
        plt.title("inpainted image")

        ax = plt.subplot(133)
        ax.imshow(compare_images(image[:,:,::-1], result, method='diff'))
        plt.title("differences between original and inpainted image")

        plt.show()