import cv2
from PIL.Image import blend
from argparse import Namespace
from os.path import join
import os
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib
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
    # Add with default arguments such as --show_plot and --output_dir
    parser = utils.create_default_argparser(output_dir = "outputs/integrated_gradients")
    
    parser.add_argument("--object_centric", action="store_true",
                        help="This flag enables the object centric version of the algorithm, " + \
                             "where we linearly interpolate between sequentially inpainted "   + \
                             "versions of the original image, i.e. objects are removed one-by-one.")

    utils.add_choices("--interpolation", choices=["linear-input", "linear-latent"], parser=parser)

    utils.add_choices("--baseline", choices=["black", "inpainted", "random"], parser=parser)

    utils.add_choices("--heatmap_type", choices=["green-red", "product", "heatmap"], parser=parser)
    
    utils.add_choices("--plot_type", choices=["single", "single+interpolation", "grid"], parser=parser)

    parser.add_argument("--grid_size", type=int, default=5,
                        help="The number of rows/columns in the grid plot," + \
                             "if 'visualization' is set to 'grid'.")

    parser.add_argument("--n_steps", type=int, default=30,
                        help="The number of steps in the integral approximation." +\
                             "If 'object_centric' is set, then this is the number of steps " + \
                             "between each inpainting.")
    
    parser.add_argument("--target_label", type=int, default=None,
                        help="The target label's index for IG. By default it is" + \
                             "the top-1 prediction of the classifier on the original image.")

    parser.add_argument("--dilation", type=int, default=7,
                        help="If given, and the baseline is inpainted, then the " + \
                             "binary mask of the instance segmentation network " + \
                             "will be diluted by this number of iterations.")

    parser.add_argument("--device", type=str, default="cpu",
                        help="The name of the device to use (e.g. 'cpu' or 'cuda').")

    args = parser.parse_args()
    
    return args

# -----------------------------------------------------------------------------------

def main():
    """
    This script calculates the integrated gradients heatmaps for all images
    in the given dataset.
    """
    progress_bar = tqdm(sorted(os.listdir(args.data_dir)))
    for idx, file in enumerate(progress_bar):
        progress_bar.set_description(file)

        image = open_pil_image(file)
        baseline = get_baseline(image)
        
        # Create interpolation
        if args.object_centric:
            interpolation_images = create_interpolation_object_centric(baseline, image)
        else:
            interpolation_images = create_interpolation(baseline, image)

        # Select the target label
        if args.target_label is None:
            target_label, prob = get_classifier_prediction(image)
        else:
            target_label, prob = args.target_label, None

        # Calculate the IG attributions
        attributions = integrated_gradients(classifier, interpolation_images, target_label)
        
        plot_results(idx, interpolation_images, attributions, target_label, prob, file)

# -----------------------------------------------------------------------------------
def open_pil_image(filename: str) -> PIL.Image.Image:
    """
    Return the given image file as an RGB PIL Image.
    """
    image_pil   = PIL.Image.open(join(args.data_dir, filename)).convert("RGB")
    
    return image_pil


def get_classifier_prediction(image: PIL.Image.Image) -> Tuple[int, float]:
    """
    Return the predicted label and its probability using the classifier.
    """
    preds = classifier.predict(image).squeeze()
    label = preds.argmax().item()
    prob = softmax(preds.detach().cpu().numpy()).max().item()

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
    Run the instance segmentation model on the given OpenCV/PIL image, and return
    the computed object mask(s) and optionally the class labels for each object.

    Args:
        image:  An image in the OpenCV or PIL format
        merge_objects:  If set, then the object masks are merged together
        return_labels:  If set, then the class labels are returned for each object
    
    Returns:
        The binary masks with the same shape as the image, and the class labels 
        if 'return_labels' is set.
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
    Create a list of images in such a way that objects are successively removed
    from the original image using inpainting, then linear interpolation is performed
    between the images with 'args.n_steps' steps. Finally, after every object is removed,
    we interpolate to the baseline image.

    NOTE: The returned list starts with the baseline and ends with the original image.

    Args:
        baseline:  The baseline image in the PIL format
        image:  The original image in the PIL format
    
    Returns:
        A list of interpolated images from baseline to the original image.
    """
    masks, labels = get_object_masks(image, merge_objects = False, return_labels = True)
    
    
    progress_bar = tqdm(
        zip(masks, labels), 
        desc="Removing objects sequentially", leave=False, total=len(masks)
    )

    # We start with the original image
    images = [image]
    for mask, label in progress_bar:
        # Remove the objects one-by-one and store the intermediate results
        cv_image = utils.pil_to_opencv_image(image)
        image = inpainting_model.inpaint(cv_image, mask)
        images.append(PIL.Image.fromarray(image))
    
    # We will interpolate from the fully inpainted image to the baseline
    images.append(baseline)
    # We expect that the interpolation goes from baseline to original image
    images.reverse()

    interpolations = []
    for idx in range(len(images) - 1):
        # Linearly interpolate between subsequent images
        interpolations += [images[idx] * (1-a) + images[idx+1] * a
                           for a in np.linspace(0, 1, num=args.n_steps)]
    
    interpolations = [PIL.Image.fromarray(np.uint8(image)) for image in interpolations]

    return interpolations

def create_interpolation(
    baseline: PIL.Image.Image, 
    image: PIL.Image.Image
) -> List[PIL.Image.Image]:
    """
    Create a linear or latent interpolation between the baseline and the original image.
    """
    if args.interpolation == "linear-input":
        interpolation_np = [
            baseline * (1-a) + image * a 
            for a in np.linspace(0, 1, num=args.n_steps, endpoint=True)
        ]

        interpolation_pil = [PIL.Image.fromarray(np.uint8(img)) for img in interpolation_np]
        
    elif args.interpolation == "linear-latent":
        interpolation_pil = generative_model.interpolate(
            baseline, image, 
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
    Return the heatmap values as computed with integrated gradients.

    Args:
        classifier:  A classification network
        interpolation_images:  A list of PIL images containing the interpolation 
                               from the baseline to the original image
        label:  The target class to condition on

    Returns:
        A tensor of shape (H,W,C) containing the heatmap values.
    """
    n_steps = len(interpolation_images)
    sum_gradients = torch.zeros(3, 224, 224)
    preprocess = lambda img : classifier.dataset.val_transforms_img(img)
    for img in tqdm(interpolation_images, desc="Calculating integrated gradients", leave=False):
        semantic_mask, semantic_scores = classifier.get_segmentation(img)
        image = preprocess(img).to(args.device)
        semantic_mask = semantic_mask.to(args.device)
        semantic_scores = semantic_scores.to(args.device)
        pred = classifier.predict_from_tensors(
            image, semantic_mask, semantic_scores, track_image_gradients=True)
        
        pred.squeeze_()

        classifier.model.zero_grad()
        pred[label].backward()
        gradient = image.grad.detach().squeeze().cpu()

        sum_gradients += gradient

    sum_gradients /= n_steps
    diff = (preprocess(interpolation_images[-1]) - preprocess(interpolation_images[0]))
    
    attributions = (diff * sum_gradients).permute(1,2,0)
    
    return attributions

def plot_results(image_idx, interpolation_images, sum_grads, label, prob, file):
    """
    Plot the attribution results as configured in 'args'.
    """
    # TODO(RN) minor refactor: its not necessary to do the entire val_transforms 
    #          it would be enough to crop the image, but we need the to_img function below
    #          for the gradients.
    if args.plot_type == "single":
        # We only plot one image and save it afterwards
        attribution_axis = plt.gca()
        skip_saving = False

    elif args.plot_type == "grid":
        global g_plot_axes
        # This is a batched version of "single"
        # Create a new figure for the first image and when the grid got full
        if image_idx % args.grid_size ** 2 == 0:
            _, g_plot_axes = plt.subplots(
                args.grid_size, args.grid_size, 
                constrained_layout=True, 
                figsize=(args.grid_size**2, args.grid_size**2))
            
        # Find the row and column for the current image
        attribution_axis = g_plot_axes[image_idx // args.grid_size, image_idx % args.grid_size]
        
        skip_saving = False if (image_idx+1) % args.grid_size**2 == 0 else True
    
    elif args.plot_type == "single+interpolation":
        # Create two-row figure of the attribution overlay and the interpolation
        fig = plt.figure(constrained_layout=True)
        n_images = len(interpolation_images)
        grid_spec = GridSpec(2, n_images, figure=fig)
        attribution_axis = fig.add_subplot(grid_spec[0, :])
        interpolation_axes = [fig.add_subplot(grid_spec[1, i]) for i in range(n_images)]
        
        skip_saving = False
    else:
        print(f"ERROR: unknown visualization type {args.plot_type}!")
        exit(-1)

    attribution_axis.axis('off')
    attribution_axis.set_title(f"{classifier.dataset.class_names[label]} ({prob*100:.0f}%)" )

    original_image = interpolation_images[-1]
    # Crop the original image to match the heatmap
    image_tensor = classifier.dataset.val_transforms_img(original_image)
    # Revert the normalization and put channel axis last
    mean = torch.Tensor(classifier.dataset.mean)
    STD = torch.Tensor(classifier.dataset.STD)
    image_tensor = image_tensor.permute(1,2,0) * STD + mean
    
    # Sum along the channel axis
    attributions = sum_grads.sum(-1, keepdim=True)
    # Normalize to so that the biggest absolute value is 1
    attributions = attributions / attributions.abs().max()
    
    plot_IG_attributions(attribution_axis, image_tensor, attributions)

    if args.plot_type == "single+interpolation":
        utils.plot_image_grid(interpolation_images, axes=interpolation_axes)

    if skip_saving:
        return

    if args.show_plot:
        plt.show()
    else:
        plt.savefig(join(args.output_dir, file), bbox_inches="tight", dpi=100)

def plot_IG_attributions(axis: Axes, original_image: torch.Tensor, attributions: torch.Tensor):
    """
    Plot the IG heatmap on the given axis. The exact visualization depends on 'args'.
    """
    if args.heatmap_type == "product":
        # Plot abs(gradients) * image        
        ig_results = original_image * attributions.abs()
        axis.imshow(ig_results)
    
    elif args.heatmap_type == "green-red":
        # Plot negative attributions in red and positive attributions in green
        attributions = torch.Tensor([
            # Red channel for negative attributions
            np.where(attributions < 0, -attributions, 0), 
            # Green channel for positive attributions
            np.where(attributions > 0, attributions, 0), 
            # Blue channel is unused
            np.zeros_like(attributions)
        ])
        attributions = attributions.squeeze(-1).permute(1,2,0)
        
        # Overlaid on the greyscale image
        to_img = lambda img : np.uint8(original_image * 255)
        greyscale_image = PIL.Image.fromarray(to_img(original_image)).convert("LA")

        axis.imshow(greyscale_image)
        axis.imshow(attributions, alpha=0.5)
        
    elif args.heatmap_type == "heatmap":
        raise NotImplementedError("'heatmap' visualization of IG is not implemented!")

    else:
        print(f"ERROR: unexpected figure type: {args.plot_type}")
        exit(-1)

if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    utils.save_args_to_output_dir(args)
    classifier = SceneRecognitionModel(
        segmentation_model = SemanticSegmentationModel(device=args.device),
        device = args.device
    )

    if args.baseline == "inpainted" or args.object_centric:
        instance_segmentation_model = InstanceSegmentationModel(device=args.device)
        inpainting_model = InpaintingModel()

    if "latent" in args.interpolation:
        generative_model = ImageGenerator(device=args.device)

    main()
    