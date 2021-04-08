import os
import PIL
from scripts import utils
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import softmax
import numpy as np
import torch
from torchvision import transforms

from models.backends.SASceneNet import utils
from models.backends.SASceneNet.SASceneNet import SASceneNet
from models.wrappers.segmentation import SemanticSegmentationModel

class PlacesDatasetMetadata:
    """
    This is a convenience class containing the number of classes in the places
    dataset and the data transformations used by the SASceneNet model.

    Args:
        do_ten_crops:  If True, perform an additional ten-crop transformation
                       (see https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.TenCrop)
    """
    def __init__(self, do_ten_crops: bool):
        self.class_names, self.class_folders = self.load_class_names()
        self.n_scene_classes = len(self.class_names)
        self.n_semantic_classes = 151
        assert self.n_scene_classes == 365

        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.output_size = 224

        self.initialize_transforms(do_ten_crops)
    
    def initialize_transforms(self, do_ten_crops: bool):
        """
        Initialize the necessary transformations for feeding an input to SASceneNet.
        """
        if do_ten_crops:
            self.val_transforms_img = transforms.Compose([
                transforms.TenCrop(self.output_size),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

            self.val_transforms_sem_mask = transforms.Compose([
                transforms.TenCrop(self.output_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [torch.from_numpy(np.asarray(crop) + 1).long().permute(2, 0, 1) for crop in crops])),
            ])

            self.val_transforms_sem_scores = transforms.Compose([
                transforms.TenCrop(self.output_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            ])
        else:
            self.val_transforms_img = transforms.Compose([
                transforms.CenterCrop(self.output_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])

        
            self.val_transforms_sem_mask = transforms.Compose([
                transforms.CenterCrop(self.output_size),
                transforms.Lambda(
                    lambda sem: torch.from_numpy(np.asarray(sem) + 1).long())
            ])

            self.val_transforms_sem_scores = transforms.Compose([
                transforms.CenterCrop(self.output_size)
            ])
        
    def load_class_names(self):
        class_names = []
        class_folders = []
        class_file_name = "../models/backends/SASceneNet/categories_places365.txt"

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                class_folders.append(line)
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]

                class_names.append(line[split_indices[1] + 1:])
        
        return class_names, class_folders

class SceneRecognitionModel:
    def __init__(self, 
        segmentation_model : SemanticSegmentationModel,
        model_path   : str = "../../utils/SAScene_checkpoint/SAScene_ResNet18_Places.pth.tar",
        device       : str = "cpu",
        do_ten_crops : bool = False
    ):
    #TODO(RN) documentation
        self.do_ten_crops = do_ten_crops
        self.dataset = PlacesDatasetMetadata(do_ten_crops)
        self.segmentation_model = segmentation_model
        self.model = SASceneNet(
            arch = "ResNet-18",
            scene_classes = self.dataset.n_scene_classes, 
            semantic_classes = self.dataset.n_semantic_classes
        )
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict_from_tensors(self,
        image: torch.Tensor,
        semantic_mask: torch.Tensor,
        semantic_scores: torch.Tensor,
        track_image_gradients: bool = False
    ) -> torch.Tensor:
        if self.do_ten_crops:
            expected_shape = (10, 3, self.dataset.output_size, self.dataset.output_size)
        else:
            expected_shape = (3, self.dataset.output_size, self.dataset.output_size)

        for tensor in (image, semantic_mask, semantic_scores):
            assert tensor.shape == expected_shape

        image.unsqueeze_(0)
        semantic_mask.unsqueeze_(0)
        semantic_scores.unsqueeze_(0)

        batch_size = image.shape[0]

        if self.do_ten_crops:
            n_crops = image.shape[1]
            # Fuse batch size and ncrops to set the input for the network
            fuse_first_two_dims = lambda t : t.reshape_(-1, t.shape[2:])

            fuse_first_two_dims(image)
            fuse_first_two_dims(semantic_mask)
            fuse_first_two_dims(semantic_scores)
            
        # Create tensor of probabilities from semantic_mask
        semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores, C=self.dataset.n_semantic_classes)
                
        if track_image_gradients:
            image.requires_grad_()

        # Model Forward
        outputSceneLabels, feature_conv, outputSceneLabelRGB, outputSceneLabelSEM = self.model(image, semanticTensor)

        if self.do_ten_crops:
            # Average results over the 10 crops
            outputSceneLabels = outputSceneLabels.view(batch_size, n_crops, -1).mean(1)
            outputSceneLabelRGB = outputSceneLabelRGB.view(batch_size, n_crops, -1).mean(1)
            outputSceneLabelSEM = outputSceneLabelSEM.view(batch_size, n_crops, -1).mean(1)

        return outputSceneLabels

    def predict(self,
        image : PIL.Image.Image,
        track_image_gradients: bool = False
    ) -> torch.Tensor:
        if image.mode is not "RGB":
            image = image.convert("RGB")
        
        semantic_mask, semantic_scores = self.get_segmentation(image)
        image = self.dataset.val_transforms_img(image)

        return self.predict_from_tensors(image, semantic_mask, semantic_scores, track_image_gradients)
        
        # if batch_size is 1:
        #     feature_conv = torch.unsqueeze(feature_conv[4, :, :, :], 0)
        #     image = torch.unsqueeze(image[4, :, :, :], 0)

        #     Ten_Predictions = utils.obtainPredictedClasses(outputSceneLabel)

        #     # Save predicted label and ground-truth label
        #     Predictions[i] = Ten_Predictions[0]
        #     SceneGTLabels[i] = sceneLabelGT.item()

        #     # Compute activation maps
        #     # utils.saveActivationMap(model, feature_conv, Ten_Predictions, sceneLabelGT,
        #     #                         image, classes, i, set, save=True)
        
        # Compute class accuracy
        

    def get_segmentation(self, image):
        semantic_mask, semantic_scores = self.segmentation_model.compute_top3_segmentation_masks(image)

        semantic_mask = self.dataset.val_transforms_sem_mask(semantic_mask)
        semantic_scores = self.dataset.val_transforms_sem_scores(semantic_scores)

        return semantic_mask, semantic_scores
