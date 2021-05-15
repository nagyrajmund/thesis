import numpy as np
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from torchvision import transforms
setup_logger()

import torch
from mit_semseg.config import cfg
from mit_semseg.models import ModelBuilder, SegmentationModule

import PIL
from typing import Tuple

from scripts.utils import infer_detectron2_class_names

class SemanticSegmentationModel:
    """
    A wrapper around the UPerNet-50 semantic segmentation network, which is used
    to acquire input segmentation masks for the SASceneNet classifier.

    Args:
        #TODO(RN) documentation
    """
    def __init__(self, config_file: str = "../../utils/upernet-50.yml", device: str = "cpu"):
        cfg.merge_from_file(config_file)
        
        self.device = device
        self.model = self._construct_model().to(self.device)
        self.model.eval()
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _construct_model(self):
        if len(cfg.MODEL.weights_encoder) == 0:
            raise ValueError("The encoder weights are not configured for the (UPerNet) semantic segmentation model.\n" + \
                             "Please set the 'model.weights_encoder' to the path of the saved checkpoint!")
        
        if len(cfg.MODEL.weights_decoder) == 0:
            raise ValueError("The decoder weights are not configured for the (UPerNet) semantic segmentation model.\n" + \
                             "Please set the 'model.weights_decoder' to the path of the saved checkpoint!")
        
        net_encoder = ModelBuilder.build_encoder(
            arch    = cfg.MODEL.arch_encoder.lower(),
            fc_dim  = cfg.MODEL.fc_dim,
            weights = cfg.MODEL.weights_encoder
        ).eval()

        net_decoder = ModelBuilder.build_decoder(
            arch        = cfg.MODEL.arch_decoder.lower(),
            fc_dim      = cfg.MODEL.fc_dim,
            num_class   = cfg.DATASET.num_class,
            weights     = cfg.MODEL.weights_decoder,
            use_softmax = True
        ).eval()

        return SegmentationModule(net_encoder, net_decoder, crit=None)

    def compute_top3_segmentation_masks(self, image: PIL.Image.Image):
        # TODO(RN) support batch of images?
        image = self._img_transform(image).unsqueeze(0).to(self.device)
        feed_dict = {"img_data": image}
        probs = self.model(feed_dict, segSize=(256, 256))
        probs = probs.squeeze(0)

        semantic_scores, semantic_mask = torch.topk(probs, 3, dim=0)
        
        return semantic_mask, semantic_scores

    def _img_transform(self, img):
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy())) #TODO(RN) Why copy?
        
        return img

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
        if len(outputs) == 0:
            print("WARNING: the instance segmentation network did not find any objects.")
            pred_masks = None
            pred_classes = None
        else:
            # The instance masks of shape (N, H, W) containing bool values
            pred_masks = outputs.pred_masks
            pred_masks = np.stack(pred_masks.cpu().numpy()).astype(np.float)
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

def binary_masks_to_opencv_images(masks):
    """
    Convert a sequence of binary masks into openCV images.
    """
    masks = masks.cpu().numpy()
    masks = [ PIL.Image.fromarray(mask).convert('RGB') for mask in masks ]
    masks = np.stack(masks)
    
    # Convert to BGR
    masks = masks[:, :, :, ::-1]

    return masks