from argparse import Namespace
from os.path import join
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from models.wrappers.classification import SceneRecognitionModel
from models.wrappers.segmentation import SemanticSegmentationModel
import PIL
import torch
from tqdm import tqdm
from scripts import utils
from pprint import pprint

def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    parser = utils.create_default_argparser(output_dir = "outputs/integrated_gradients")
    
    parser.add_argument("--n_steps", type=int, default=30,
                        help="The number of steps in the integral approximation")
    
    parser.add_argument("--target_label", type=int, default=None,
                        help="The target label's index for IG. By default it is the top-1 " + \
                             "prediction of the classifier on the original image.")

    interpolation_options = ["linear-input", "linear-latent"]
    parser.add_argument("--interpolation", choices=interpolation_options, default=interpolation_options[0],
                        help=f"One of the following: {interpolation_options}.")
    
    baseline_options = ["black_image", "inpainted", "random"]
    parser.add_argument("--baseline", choices=baseline_options, default=baseline_options[0],
                        help=f"One of the following: {baseline_options}.")
    
    visualization_options = ["green-red"]
    parser.add_argument("--visualization", choices=visualization_options, default=visualization_options[0],
                        help=f"One of the following: {visualization_options}.")

    args = parser.parse_args()
    print("-"*80)
    pprint(vars(args))
    print("-"*80)

    return args

def main():
    """
    This script calculates the integrated gradients heatmaps for all images in the given dataset.
    """
    for file in os.listdir(args.data_dir):
        image = open_image(file)
        baseline = get_baseline(image)
        interpolation_images = create_interpolation(baseline, image)
        target_label = get_target_label(interpolation_images)

        attributions = integrated_gradients(classifier, interpolation_images, target_label)

        plot_results(interpolation_images[-1], attributions, target_label, file)

def open_image(filename: str) -> np.ndarray:
    image_pil   = PIL.Image.open(join(args.data_dir, filename)).convert("RGB")
    image_np    = np.asarray(image_pil) / 255
    
    return image_np

def get_target_label(interpolation_images: List[PIL.Image.Image]) -> int:
    if args.target_label is not None:
        return args.target_label
    
    return classifier.predict(interpolation_images[-1]).squeeze().argmax().item()


def get_baseline(image: np.ndarray) -> np.ndarray:
    if args.baseline == "black_image":
        baseline = np.zeros_like(image)

    elif args.baseline == "random":
        baseline = np.random.rand_like(image)

    elif args.baseline == "inpainted":
        raise NotImplementedError()

    return baseline

def create_interpolation(
    baseline: np.ndarray, 
    image: np.ndarray
) -> List[PIL.Image.Image]:
    """
    TODO(RN) documentation
    """
    interpolation_np = [
        baseline * (1-a) + image * a 
        for a in np.linspace(0, 1, num=args.n_steps, endpoint=True)
    ]

    interpolation_pil = [PIL.Image.fromarray(np.uint8(img*255)) for img in interpolation_np]
    
    return interpolation_pil


def integrated_gradients(
    classifier: SceneRecognitionModel,
    interpolation_images: List[PIL.Image.Image], 
    label: int
) -> torch.Tensor:
    """
    TODO(RN) documentation
    """
    n_steps = len(interpolation_images)
    sum_gradients = torch.zeros(3, 224, 224)

    preprocess = lambda img : classifier.dataset.val_transforms_img(img)

    for img in tqdm(interpolation_images):
        semantic_mask, semantic_scores = classifier.get_segmentation(img)
        image = classifier.dataset.val_transforms_img(img)

        pred = classifier.predict_from_tensors(
            image, semantic_mask, semantic_scores, track_image_gradients=True)
        
        pred.squeeze_()
        classifier.model.zero_grad()

        pred[label].backward()
        gradient = image.grad.detach().squeeze()

        sum_gradients += gradient

    sum_gradients /= n_steps
    diff = (preprocess(interpolation_images[-1]) - preprocess(interpolation_images[0]))

    return diff * sum_gradients

def plot_results(image, sum_grads, label, file):
    # TODO(RN) minor refactor: its not necessary to do the entire val_transforms 
    #          it would be enough to crop the image, but we need the to_img function below
    #          for the gradients.
    image = classifier.dataset.val_transforms_img(image)
    _, axes = plt.subplots(1, 2)
        
    mean = torch.Tensor(classifier.dataset.mean)
    STD = torch.Tensor(classifier.dataset.STD)

    to_img = lambda x : np.uint8((x.permute(1,2,0) * STD + mean) * 255)

    axes[0].axis('off')
    greyscale_image = PIL.Image.fromarray(to_img(image)).convert("LA")
    axes[0].imshow(greyscale_image)
    
    plt.title(classifier.dataset.class_names[label])

    multipliers = sum_grads.squeeze(0).sum(0, keepdim=True).permute(1,2,0)
    multipliers = multipliers / multipliers.max()
    
    attributions = torch.Tensor([
        # Red channel for negative attributions
        np.where(multipliers < 0, -multipliers, 0), 
        # Green channel for positive attributions
        np.where(multipliers > 0, multipliers, 0), 
        # Blue channel is unused
        np.zeros_like(multipliers)
    ])

    axes[0].imshow(attributions.squeeze(-1).permute(1,2,0), alpha=0.5)
    
    ig = np.uint8((image.permute(1,2,0) * STD + mean) * 255 * np.where(multipliers > 0, multipliers, 0))
    
    axes[1].axis('off')
    axes[1].imshow(ig)

    plt.savefig(join(args.output_dir, file), bbox_inches="tight")

if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    segmentation_net = SemanticSegmentationModel()
    classifier = SceneRecognitionModel(segmentation_net)
    
    main()
    