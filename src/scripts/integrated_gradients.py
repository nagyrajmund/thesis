from argparse import Namespace
from os.path import join
import os
from matplotlib import pyplot as plt
import numpy as np
from models.wrappers.classification import SceneRecognitionModel
from models.wrappers.segmentation import SemanticSegmentationModel
import PIL
import torch
from tqdm import tqdm
from scripts import utils

def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    parser = utils.create_default_argparser(output_dir = "outputs/integrated_gradients")
    parser.add_argument("--n_steps", type=int, default=30,
                        help="The number of steps in the integral approximation")
    args = parser.parse_args()
    return args

def main():
    for file in os.listdir(args.data_dir):
        image = PIL.Image.open(join(args.data_dir, file)).convert("RGB")
        
        image_np = np.asarray(image) / 255
        baseline_np = np.zeros_like(image_np)
        images = [image_np * a + baseline_np * (1-a) for a in np.linspace(0, 1, num=args.n_steps, endpoint=True)]
        images = [PIL.Image.fromarray(np.uint8(img*255)) for img in images]
        
        label = model.predict(images[-1]).squeeze().argmax().item()
        sum_grads = integrated_gradients(model, images, label)

        plot_results(images[-1], sum_grads, label, file)


def integrated_gradients(model: SceneRecognitionModel, interpolation_images, label):
    n_steps = len(interpolation_images)
    sum_gradients = torch.zeros(3, 224, 224)

    preprocess = lambda img : model.dataset.val_transforms_img(img)

    for img in tqdm(interpolation_images):
        if img.endswith(".jpg"):
            semantic_mask, semantic_scores = model.get_segmentation(img)
            image = model.dataset.val_transforms_img(img)

            pred = model.predict_from_tensors(
                image, semantic_mask, semantic_scores, track_image_gradients=True)
            
            pred.squeeze_()
            model.model.zero_grad()

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
    image = model.dataset.val_transforms_img(image)
    _, axes = plt.subplots(1, 2)
        
    mean = torch.Tensor(model.dataset.mean)
    STD = torch.Tensor(model.dataset.STD)

    to_img = lambda x : np.uint8((x.permute(1,2,0) * STD + mean) * 255)

    axes[0].axis('off')
    greyscale_image = PIL.Image.fromarray(to_img(image)).convert("LA")
    axes[0].imshow(greyscale_image)
    
    plt.title(model.dataset.class_names[label])

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
    model = SceneRecognitionModel(segmentation_net)
    
    main()
    