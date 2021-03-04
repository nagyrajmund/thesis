import cv2
import torch
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt

class SegmentationModel:
    """
    A wrapper around pretrained instance segmentation models from the Detectron2 model zoo.

    Args:
        model_config:   The name of the model's config file in the model zoo
        device:         The device of the model ('cpu' or e.g. 'cuda')
        threshold:      The minimum threshold for accepting class predictions as positive
    """
    def __init__(self,
        model_config : str   = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        device       : str   = "cpu",
        threshold    : float = 0.3
    ):
        self.threshold = threshold

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.DEVICE = device
        
        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get("coco_2017_train").thing_classes

    def extract_segmentation(self,
        image
    ):
        """
        Args:
            image:  An image of shape (H, W, C) in BGR order
        
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

a = SegmentationModel()
image = cv2.imread("data/places_small/Places365_val_00000178.jpg")

masks, labels = a.extract_segmentation(image)

for mask, label in zip(masks, labels):
    plt.imshow(mask)
    plt.title(a.class_names[label])
    plt.show()