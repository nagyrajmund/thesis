import numpy as np
import cv2
import torch
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()

import PIL
from skimage.util import compare_images
import matplotlib.pyplot as plt
from typing import Tuple

from inpainting import InpaintingModel
from src.utils import infer_detectron2_class_names

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
        
        config = detectron2.config.get_cfg()
        config.merge_from_file(model_zoo.get_config_file(config_file))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        config.MODEL.DEVICE = device
        
        self.predictor = DefaultPredictor(config)

    def extract_segmentation(self,
        image: np.ndarray
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Args:
            image:  An image of shape (H, W, C) in BGR order (that openCV uses by default)
        
        Returns:
            pred_masks:     the binary instance segmentation masks
            pred_classes:   the corresponding class labels
        """
        outputs = self.predictor(image)["instances"]
    
        # The instance masks of shape (N, H, W) containing bool values
        pred_masks = outputs.pred_masks
        # The vector of N class labels
        pred_classes = outputs.pred_classes
        
        return pred_masks, pred_classes

#TODO(RN) remove
if __name__ == "__main__":
    inpainting_model = InpaintingModel()
    segmentation_model = SegmentationModel(threshold=0.5)
    image = cv2.imread("../data/places_small/Places365_val_00000173.jpg")
    masks, labels = segmentation_model.extract_segmentation(image)
    
    for mask, label in zip(masks, labels):
        # Create RGB Pillow Image from binary mask
        mask = PIL.Image.fromarray(mask.numpy()).convert('RGB')
        # Convert to BGR as openCV expects it
        mask = np.asarray(mask)[::-1]

        result = inpainting_model.inpaint(image, mask)

        ax = plt.subplot(131)
        ax.imshow(image[:,:,::-1])
        ax.imshow(mask, alpha = 0.2)
        plt.title("original image and mask ({})".format(
            segmentation_model.class_names[label])
        )

        ax = plt.subplot(132)
        ax.imshow(result[0])
        plt.title("inpainted image")

        ax = plt.subplot(133)
        ax.imshow(compare_images(image[:,:,::-1], result[0], method='diff'))
        plt.title("differences between original and inpainted image")

        plt.show()