import cv2
from PIL.Image import blend
from argparse import Namespace
from os.path import join
import os
from typing import List, Union
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from models.wrappers.classification import SceneRecognitionModel
from models.wrappers.segmentation import InstanceSegmentationModel, SemanticSegmentationModel
from models.wrappers.generation import ImageGenerator
from models.wrappers.inpainting import InpaintingModel
import PIL
import torch
from tqdm import tqdm
from scripts import utils
from pprint import pprint
from scipy.special import softmax

def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    parser = utils.create_default_argparser(output_dir = "outputs/integrated_gradients")
    
    parser.add_argument("--object_centric", action="store_true")

    utils.add_choices("--interpolation", choices=["linear-input", "linear-latent"], parser=parser)

    utils.add_choices("--baseline", choices=["black", "inpainted", "random"], parser=parser)

    utils.add_choices("--colors", choices=["green-red", "black-white"], parser=parser)
    
    utils.add_choices("--visualization", choices=["single", "single+interpolation", "grid"], parser=parser)

    parser.add_argument("--grid_size", type=int, default=5,
                        help="The number of rows/columns in the grid plot," + \
                             "if 'visualization' is set to \"grid\".")

    parser.add_argument("--n_steps", type=int, default=30,
                        help="The number of steps in the integral approximation")
    
    parser.add_argument("--target_label", type=int, default=None,
                        help="The target label's index for IG. By default it is" + \
                             "the top-1 prediction of the classifier on the original image.")

    parser.add_argument("--dilation", type=int, default=7,
                        help="If given, and the baseline is inpainted, then the" + \
                             " binary mask of the instance segmentation network" + \
                             " will be diluted by this number of iterations.")

    args = parser.parse_args()
    
    print("-"*80)
    pprint(vars(args))
    print("-"*80)

    return args

# -----------------------------------------------------------------------------------

def main():
    """
    This script calculates the integrated gradients heatmaps for all images
    in the given dataset.
    """
    progress_bar = tqdm(sorted(os.listdir(args.data_dir)))
    for i, file in enumerate(progress_bar):
        progress_bar.set_description(file)

        image = open_pil_image(file)
        baseline = get_baseline(image)

        if args.object_centric:
            interpolation_images = create_interpolation_object_centric(baseline, image)
        else:
            interpolation_images = create_interpolation(baseline, image)

        if args.target_label is None:
            target_label, prob = get_classifier_prediction(file)
        else:
            target_label = args.target_label
            prob = None

        attributions = integrated_gradients(
            classifier, interpolation_images, target_label
        )
        
        if args.visualization == "grid":
            
            # Create subplot
            if i % args.grid_size ** 2 == 0:
                _, axes = plt.subplots(args.grid_size, args.grid_size)

            axis = axes[i // args.grid_size, i % args.grid_size]
            
            skip_saving = False if (i+1) % args.grid_size**2 == 0 else True
        
        elif args.visualization == "single" or args.visualization == "single+interpolation":
            axis = None
            skip_saving = False

        else:
            print(f"ERROR: unknown visualization type {args.visualization}!")
            exit(-1)

        plot_results(interpolation_images, attributions, target_label, prob, file, axis, skip_saving)

# -----------------------------------------------------------------------------------
def open_pil_image(filename: str) -> PIL.Image.Image:
    """
    Return the given image file as an RGB PIL Image.
    """
    image_pil   = PIL.Image.open(join(args.data_dir, filename)).convert("RGB")
    
    return image_pil


def get_classifier_prediction(file: str) -> int:
    image = open_pil_image(file)
    
    preds = classifier.predict(image).squeeze()
    label = preds.argmax().item()
    prob = softmax(preds.detach().numpy()).max().item()

    return label, prob


def get_baseline(image: PIL.Image.Image) -> np.ndarray:
    """
    Return the IG baseline corresponding to the given image file.
    """
    if args.baseline == "black":
        baseline = np.zeros_like(np.asarray(image), dtype=np.uint8)

    elif args.baseline == "random":
        shape = np.asarray(image).shape
        # Fill with Gaussian noise
        baseline = np.random.randn(*shape)
        baseline = np.uint8(baseline * 255)

    elif args.baseline == "inpainted":
        cv_image = utils.pil_to_opencv_image(image)
        mask = get_object_masks(cv_image, merge_objects = True).squeeze()
        # TODO(RN) is mask right?
        baseline = inpainting_model.inpaint(cv_image, mask)

    else:
        print(f"ERROR: unexpected baseline '{args.baseline}'!")
        exit(-1)

    return PIL.Image.fromarray(baseline)

def get_object_masks(
    image: Union[np.ndarray, PIL.Image.Image],
    merge_objects: bool,
    return_labels: bool = False
) -> np.ndarray:
    """
    TODO(RN) documentation
    """
    # The instance segmentation model expects OpenCV images
    if isinstance(image, PIL.Image.Image):
        image = utils.pil_to_opencv_image(image)

    if return_labels:
        masks, labels = instance_segmentation_model.extract_segmentation(image, return_labels=True)
    else:
        masks = instance_segmentation_model.extract_segmentation(image, return_labels=False)

    if merge_objects:
        masks = masks.max(axis = 0, keepdims = True)

    if args.dilation > 0:
        for i in range(len(masks)):
            masks[i] = utils.dilate_mask(masks[i], n_iters=args.dilation)

    if return_labels:
        return masks, labels
    else:
        return masks

def create_interpolation_object_centric(
    baseline: PIL.Image,
    image: PIL.Image
) -> List[PIL.Image.Image]:
    """
    TODO(RN) documentation
    """
    masks, labels = get_object_masks(image, merge_objects = False, return_labels = True)
    
    # The interpolation should be from baseline to original image, but we build it
    # from original image to baseline instead, and reverse it later.
    # This is logical because we keep removing objects from the original instead of
    #  adding objects to the baseline.
    interpolation = [image]

    progress_bar = tqdm(
        zip(masks, labels), 
        desc="Removing objects sequentially", leave=False, total=len(masks)
    )
    
    for mask, label in progress_bar:
        # In each iteration we remove one more object
        cv_image = utils.pil_to_opencv_image(image)
        image = inpainting_model.inpaint(cv_image, mask)
        interpolation.append(PIL.Image.fromarray(image))
    
    interpolation.append(baseline)
    interpolation.reverse()

    return interpolation

def create_interpolation(
    baseline: np.ndarray, 
    image: np.ndarray
) -> List[PIL.Image.Image]:
    """
    TODO(RN) documentation
    """

    if args.interpolation == "linear-input":
        interpolation_np = [
            baseline * (1-a) + image * a 
            for a in np.linspace(0, 1, num=args.n_steps, endpoint=True)
        ]

        interpolation_pil = [PIL.Image.fromarray(np.uint8(img*255)) for img in interpolation_np]


    elif args.interpolation == "linear-latent":
        baseline_pil = PIL.Image.fromarray(np.uint8(baseline * 255))
        image_pil = PIL.Image.fromarray(np.uint8(image * 255))

        interpolation_pil = generative_model.interpolate(
            baseline_pil, image_pil, 
            n_steps = args.n_steps, 
            add_original_images=False
        )

    else:
        print(f"ERROR: unexpected interpolation type: {args.interpolation}")
        exit(-1)

    
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

    for img in tqdm(interpolation_images, desc="Calculating integrated gradients", leave=False):
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

def plot_results(interpolation_images, sum_grads, label, prob, file, axis=None, skip_saving=False):
    # TODO(RN) minor refactor: its not necessary to do the entire val_transforms 
    #          it would be enough to crop the image, but we need the to_img function below
    #          for the gradients.
    if args.visualization == "single+interpolation":
        fig = plt.figure(constrained_layout=True)
        grid_spec = GridSpec(2, args.n_steps, figure=fig)
        interpolation_axes = [fig.add_subplot(grid_spec[1, i]) for i in range(args.n_steps)]
        axis = fig.add_subplot(grid_spec[0, :])
    elif axis is None:
        axis = plt.gca() 

    axis.axis('off')
    axis.set_title(f"{classifier.dataset.class_names[label]} ({prob*100:.0f}%)" )

    original_image = interpolation_images[-1]
    original_image = classifier.dataset.val_transforms_img(original_image)
        
    attributions = sum_grads.squeeze(0).sum(0, keepdim=True).permute(1,2,0)
    attributions = attributions / attributions.abs().max()
    
    plot_IG_attributions(axis, original_image, attributions)

    if args.visualization == "single+interpolation":
        utils.plot_image_grid(interpolation_images, axes=interpolation_axes)

    if skip_saving:
        return

    elif args.show_plot:
        plt.show()
    else:
        plt.savefig(join(args.output_dir, file), bbox_inches="tight")

def plot_IG_attributions(axis, original_image, attributions):
    mean = torch.Tensor(classifier.dataset.mean)
    STD = torch.Tensor(classifier.dataset.STD)
    to_img = lambda x : np.uint8((x.permute(1,2,0) * STD + mean) * 255)

    if args.colors == "black-white":
        unnormalized_image = to_img(original_image)
        ig_results = unnormalized_image * np.max(attributions, 0)
        axis.imshow(ig_results)
    
    elif args.colors == "green-red":
        greyscale_image = PIL.Image.fromarray(to_img(original_image)).convert("LA")

        attributions = torch.Tensor([
            # Red channel for negative attributions
            np.where(attributions < 0, -attributions, 0), 
            # Green channel for positive attributions
            np.where(attributions > 0, attributions, 0), 
            # Blue channel is unused
            np.zeros_like(attributions)
        ])

        attributions = attributions.squeeze(-1).permute(1,2,0)

        axis.imshow(greyscale_image)
        axis.imshow(attributions, alpha=0.5)
        
    
    else:
        print(f"ERROR: unexpected visualization type: {args.visualization}")
        exit(-1)

if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    
    classifier = SceneRecognitionModel(
        segmentation_model = SemanticSegmentationModel()
    )

    if args.baseline == "inpainted" or args.object_centric:
        instance_segmentation_model = InstanceSegmentationModel()
        inpainting_model = InpaintingModel()

    if "latent" in args.interpolation:
        generative_model = ImageGenerator()
    
    main()
    