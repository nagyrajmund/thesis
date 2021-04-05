import numpy as np
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
setup_logger()

import PIL
from typing import Tuple

from scripts.utils import infer_detectron2_class_names

def binary_masks_to_opencv_images(masks):
    """
    Convert a sequence of binary masks into openCV images.
    """
    masks = masks.numpy()
    masks = [ PIL.Image.fromarray(mask).convert('RGB') for mask in masks ]
    masks = np.stack(masks)
    
    # Convert to BGR
    masks = masks[:, :, :, ::-1]

    return masks

class InstanceSegmentationModel:
    """
    A wrapper around pretrained instance segmentation models from the Detectron2 model zoo.

    Args:
        model_config:   The name of the model's config file in the model zoo
        device:         The device of the model ('cpu' or e.g. 'cuda')
        threshold:      The minimum threshold for accepting class predictions as positive
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
            
            (if return_labels is True)
            pred_classes:   the corresponding class labels of shape (N)
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