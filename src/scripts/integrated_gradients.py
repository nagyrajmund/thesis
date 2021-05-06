import math
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
# -----------------------------------------------------------------------------------

def main():
    """
    This script calculates the integrated gradients heatmaps for all images
    in the given dataset.
    """
    args = parse_args()
    utils.create_output_dir(args)
    utils.save_args_to_output_dir(args)
    
    explainer = IntegratedGradients(args)
    progress_bar = tqdm(sorted(os.listdir(args.data_dir)))
    for idx, file in enumerate(progress_bar):
        progress_bar.set_description(file)

        # Open the input image    
        image = PIL.Image.open(join(args.data_dir, file)).convert("RGB")
    
        # Create the baseline image and the interpolation    
        baseline = explainer.get_baseline(image)
        interpolation = explainer.create_interpolation(baseline, image)

        # Select the label to condition the explanation on
        if args.target_label is None:
            target_label, prob = explainer.get_classifier_prediction(image)
        else:
            target_label, prob = args.target_label, None

        # Compute the attribution heatmap
        attributions = explainer.compute_attributions(interpolation, target_label)
        
        # Compute the metrics of the heatmap
        explainer.insertion_deletion_metrics(image, attributions)
        
        # Visualize the results
        explainer.plot_results(idx, interpolation, attributions, target_label, prob, file)

# -----------------------------------------------------------------------------------


        

    # Insertion score: start from blurred image and insert pixels

    # Deletion score: delete pixels with constant values


# ---------------------------------------------------------------------------------------


class IntegratedGradients:
    """
    A flexible class for creating specific types of Integrated Gradients explanations.

    NOTE: the keyword args below are sometimes optional, depending on the configuration 
    in 'args'. For example, the generative model is only needed with latent interpolation.
    
    Args:
        args:       A Namespace with the parameters from 'self.add_argparse_args()'
    """
    def __init__(self,
        args: Namespace,
    ):
        self.classifier = SceneRecognitionModel(
            segmentation_model = SemanticSegmentationModel(device=args.device),
            device = args.device
        )

        if args.baseline == "inpainted" or args.object_centric:
            self.instance_segmentation_model = InstanceSegmentationModel(device=args.device)
            self.inpainting_model = InpaintingModel()
        
        if "latent" in args.interpolation:
            generative_model = ImageGenerator(device=args.device)
        
        self.args = args
    
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--object_centric", action="store_true",
                        help="This flag enables the object centric version of the algorithm, " + \
                             "where we linearly interpolate between sequentially inpainted "   + \
                             "versions of the original image, i.e. objects are removed one-by-one.")

        utils.add_choices("--interpolation", parser=parser,  choices=["linear-input", "linear-latent"])

        utils.add_choices("--baseline",      parser=parser,  choices=["black", "inpainted", "random"])

        utils.add_choices("--heatmap_type",  parser=parser,  choices=["green-red", "product", "heatmap"])
        
        utils.add_choices("--plot_type",     parser=parser,  choices=["single", "single+interpolation", "grid"])

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

        parser.add_argument("--heatmap_top_pct", type=float, default=100,
                            help="This number sets the treshold for showing only the top n percent attributions.")
        return parser

    def create_interpolation(self, 
        baseline: PIL.Image.Image, 
        image: PIL.Image.Image
    ) -> List[PIL.Image.Image]:
        """
        Return the interpolation from the baseline to the original image.
        """
        if self.args.object_centric:
            return self._create_interpolation_object_centric(baseline, image)
        else:
            return self._create_interpolation(baseline, image)

    def insertion_deletion_metrics(self,
        original_image: PIL.Image.Image, 
        attributions: torch.Tensor,
        n_bins: int = 100
    ) -> Tuple[List[float], List[float]]:
        """
        Given an image and the IG heatmap, compute the probability of the target
        class as important pixels are removed/added.
        """
        attributions.squeeze_().abs_()    # TODO(RN) take abs of heatmap values?

        # Crop the original image to match the heatmap
        original_image = self.classifier.dataset.val_transforms_img(original_image)
        
        # Revert the normalization
        mean = torch.Tensor(self.classifier.dataset.mean)
        STD = torch.Tensor(self.classifier.dataset.STD)
        original_image = (original_image.permute(1,2,0) * STD + mean).permute(2,0,1)
        
        # Get the ordered indices of the flattened image
        ordered_heatmap_idxs = np.argsort(attributions.flatten())
        # Convert them back to 2D indices
        ordered_heatmap_idxs = np.dstack(np.unravel_index(ordered_heatmap_idxs, shape=attributions.shape))[0]
        blurry_image = utils.blur_image(original_image)

        real_heatmap = cv2.applyColorMap(np.uint8(255 * attributions.abs()), cv2.COLORMAP_JET)
        real_heatmap = cv2.cvtColor(real_heatmap, cv2.COLOR_BGR2RGB) / 255
        curr_heatmap = np.zeros_like(real_heatmap)
        
        n_pixels = len(attributions.flatten())
        step = math.ceil(n_pixels / n_bins)
        progress_bar = tqdm(reversed(range(0, n_pixels, step)), desc="Computing insertion score", leave=False)
        for i in progress_bar:
            selected_pixels = ordered_heatmap_idxs[i:i + step]
            
            # Restore the most important pixels
            for x, y in selected_pixels:
                blurry_image[:, x, y] = original_image[:, x, y]
                curr_heatmap[x, y, :] = real_heatmap[x, y, :]
            
            # _, axes = plt.subplots(1, 3)
            
            # for ax, img in zip(axes, [blurry_image, curr_heatmap, real_heatmap]):
            #     ax.axis('off')
            #     if img.shape[0] < 4:
            #         ax.imshow(img.permute(1,2,0))
            #     else:
            #         ax.imshow(img)
            #     ax.set_title(i)
            # plt.show()

    def plot_results(self, image_idx, interpolation_images, attributions, label, prob, file):
        """
        Plot the attribution results as configured in 'args'.
        """
        # TODO(RN) minor refactor: its not necessary to do the entire val_transforms 
        #          it would be enough to crop the image, but we need the to_img function below
        #          for the gradients.
        if self.args.plot_type == "single":
            # Plot just one image and save it afterwards
            attribution_axis = plt.gca()
            skip_saving = False

        elif self.args.plot_type == "grid":
            # 

            # the Axes are stored in a global variable because we want to keep them between function calls
            global g_plot_axes
            
            # Create a new figure for the first image and also when the current
            # figure is filled completely
            if image_idx % self.args.grid_size ** 2 == 0:
                _, g_plot_axes = plt.subplots(
                    self.args.grid_size, self.args.grid_size, 
                    constrained_layout=True, 
                    figsize=(self.args.grid_size**2, self.args.grid_size**2))
                
            # Find the row and column for the current image
            attribution_axis = g_plot_axes[image_idx // self.args.grid_size, image_idx % self.args.grid_size]
            
            skip_saving = False if (image_idx+1) % self.args.grid_size**2 == 0 else True
        
        elif self.args.plot_type == "single+interpolation":
            # Create two-row figure of the attribution overlay and the interpolation
            fig = plt.figure(constrained_layout=True)
            n_images = len(interpolation_images)
            grid_spec = GridSpec(2, n_images, figure=fig)
            attribution_axis = fig.add_subplot(grid_spec[0, :])
            interpolation_axes = [fig.add_subplot(grid_spec[1, i]) for i in range(n_images)]
            
            skip_saving = False
        else:
            print(f"ERROR: unknown visualization type {self.args.plot_type}!")
            exit(-1)

        attribution_axis.axis('off')
        attribution_axis.set_title(f"{self.classifier.dataset.class_names[label]} ({prob*100:.0f}%)" )

        original_image = interpolation_images[-1]
        # Crop the original image to match the heatmap
        image_tensor = self.classifier.dataset.val_transforms_img(original_image)
        # Revert the normalization and put channel axis last
        mean = torch.Tensor(self.classifier.dataset.mean)
        STD = torch.Tensor(self.classifier.dataset.STD)
        image_tensor = image_tensor.permute(1,2,0) * STD + mean

        
        self.plot_IG_attributions(attribution_axis, image_tensor, attributions)

        if self.args.plot_type == "single+interpolation":
            utils.plot_image_grid(interpolation_images, axes=interpolation_axes)

        if skip_saving:
            return

        if self.args.show_plot:
            plt.show()
        else:
            plt.savefig(join(self.args.output_dir, file), bbox_inches="tight", dpi=200, transparent=True)

    def plot_IG_attributions(self, axis: Axes, original_image: torch.Tensor, attributions: torch.Tensor):
        """
        Plot the IG heatmap on the given axis. The exact visualization depends on 'args'.
        """
        if self.args.heatmap_type == "product":
            # This visualization is essentially abs(gradients) * image
            multipliers = attributions.abs()
            # Threshold the heatmap values using 'self.args.heatmap_top_pct'
            raise NotImplementedError("this is buggy below, use --heatmap_type heatmap")
            threshold = np.percentile(multipliers, 100 - self.args.heatmap_top_pct)
            zeros = torch.zeros_like(multipliers).float()
            multipliers = np.where(multipliers < threshold, zeros, multipliers)
            
            ig_results = original_image * multipliers
            axis.imshow(ig_results)
        
        elif self.args.heatmap_type == "green-red":
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
            
            # Threshold the red-green channels using 'self.args.heatmap_top_pct'
            for channel in [0,1]:
                raise NotImplementedError("this is buggy below, use --heatmap_type heatmap")
                threshold = np.percentile(attributions[channel].numpy(), 100 - self.args.heatmap_top_pct)
                zeros = torch.zeros_like(attributions[channel]).float()
                attributions[channel] = torch.where(attributions[channel] < threshold, 
                    zeros, attributions[channel])
            
            # Overlaid on the greyscale image
            to_img = lambda img : np.uint8(original_image * 255)
            greyscale_image = PIL.Image.fromarray(to_img(original_image)).convert("LA")

            axis.imshow(greyscale_image)
            axis.imshow(attributions, alpha=0.5)
            
        elif self.args.heatmap_type == "heatmap":
            attributions = attributions.abs()
            # Threshold the heatmap values using 'self.args.heatmap_top_pct'
            threshold = np.percentile(attributions, 100 - self.args.heatmap_top_pct)
            zero = torch.FloatTensor([0])
            attributions = torch.where(attributions < threshold, zero, attributions)
    
            heatmap = cv2.applyColorMap(np.uint8(255 * attributions), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255
            attributions.unsqueeze_(-1)
            # axis.imshow(original_image)
            axis.imshow(attributions * heatmap + (1 - attributions) * original_image)
            
        else:
            print(f"ERROR: unexpected figure type: {self.args.plot_type}")
            exit(-1)


    def _create_interpolation(self,
        baseline: PIL.Image.Image, 
        image: PIL.Image.Image
    ) -> List[PIL.Image.Image]:
        """
        Create a linear or a latent interpolation between the baseline and the original image.
        """
        if self.args.interpolation == "linear-input":
            interpolation_np = [
                baseline * (1-a) + image * a 
                for a in np.linspace(0, 1, num=self.args.n_steps, endpoint=True)
            ]

            interpolation_pil = [PIL.Image.fromarray(np.uint8(img)) for img in interpolation_np]
            
        elif self.args.interpolation == "linear-latent":
            interpolation_pil = self.generative_model.interpolate(
                baseline, image, 
                n_steps = self.args.n_steps, 
                add_original_images=False
            )

        else:
            print(f"ERROR: unexpected interpolation type: {self.args.interpolation}")
            exit(-1)

        
        return interpolation_pil

    def _create_interpolation_object_centric(self,
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
        masks, labels = self._get_object_masks(image, merge_objects = False, return_labels = True)
        
        
        progress_bar = tqdm(
            zip(masks, labels), 
            desc="Removing objects sequentially", leave=False, total=len(masks)
        )

        # We start with the original image
        images = [image]
        for mask, label in progress_bar:
            # Remove the objects one-by-one and store the intermediate results
            cv_image = utils.pil_to_opencv_image(image)
            image = self.inpainting_model.inpaint(cv_image, mask)
            images.append(PIL.Image.fromarray(image))
        
        # We will interpolate from the fully inpainted image to the baseline
        images.append(baseline)
        # We expect that the interpolation goes from baseline to original image
        images.reverse()

        interpolations = []
        for idx in range(len(images) - 1):
            # Linearly interpolate between subsequent images
            interpolations += [images[idx] * (1-a) + images[idx+1] * a
                            for a in np.linspace(0, 1, num=self.args.n_steps)]
        
        interpolations = [PIL.Image.fromarray(np.uint8(image)) for image in interpolations]

        return interpolations
        
    def compute_attributions(self,
        interpolation_images: List[PIL.Image.Image], 
        target_label: int
    ) -> torch.Tensor:
        """
        Return the heatmap values as computed with integrated gradients.

        Args:
            classifier:            A classification network
            interpolation_images:  A list of PIL images containing the interpolation
                                from the baseline to the original image
            label:                 The index of the target class to condition on

        Returns:
            attributions:          a tensor of shape (H,W,C) containing the normalized pixel-wise attributions.
        """
        n_steps = len(interpolation_images)
        
        sum_gradients = torch.zeros(3, 224, 224)
        progress_bar = tqdm(interpolation_images, desc="Calculating integrated gradients", leave=False)
        for img in progress_bar:
            gradients = self._get_gradients_of_classifier_output_wrt_input_image(img, target_label)
            sum_gradients += gradients

        first_input = self.classifier.preprocess_img(interpolation_images[0])
        last_input = self.classifier.preprocess_img(interpolation_images[-1])

        # IG equation: (x_n        - x_1)         * sum_{i=1}^{n}{gradients_i} * (1 / n)
        attributions = (last_input - first_input) * sum_gradients              / n_steps
        # Put channel axis last 
        attributions = attributions.permute(1,2,0)
        # Sum along the channel axis
        attributions = attributions.sum(-1, keepdim=True)
        # Normalize to so that the biggest absolute value is 1
        attributions = attributions / attributions.abs().max()

        return attributions

    def _get_gradients_of_classifier_output_wrt_input_image(self,
        image: PIL.Image.Image, 
        target_label: int
    ) -> torch.Tensor:
        """
        Return the gradients of the classifier's output w.r.t. the input image.

        Args:
            image:          A PIL image of size (H,W) with C = 3 channels
            target_label:   The index of the class that the explanation is conditioned on

        Returns:
            gradients:      A tensor of shape (3,H,W) containing the gradients of the image
        """
        # Get segmentation mask, preprocess the inputs and put them on the right device
        semantic_mask, semantic_scores = self.classifier.get_segmentation(image)
        image_tensor = self.classifier.preprocess_img(image).to(self.args.device)
        semantic_mask = semantic_mask.to(self.args.device)
        semantic_scores = semantic_scores.to(self.args.device)

        # Get the classifiers prediction while tracking the gradients
        prediction = self.classifier.predict_from_tensors(
            image_tensor, semantic_mask, semantic_scores, track_image_gradients=True).squeeze_()

        # Compute and return the gradients of the image tensor
        self.classifier.model.zero_grad()
        prediction[target_label].backward()
        gradients = image_tensor.grad.detach().squeeze().cpu()

        return gradients

    def get_baseline(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Return the IG baseline corresponding to the given image file.
        """
        baseline_type = self.args.baseline
        if baseline_type == "black":
            baseline = np.zeros_like(np.asarray(image), dtype=np.uint8)

        elif baseline_type == "random":
            shape = np.asarray(image).shape
            # Fill with Gaussian noise
            baseline = np.random.randn(*shape)
            baseline = np.uint8(baseline * 255)

        elif baseline_type == "inpainted":
            cv_image = utils.pil_to_opencv_image(image)
            mask = self._get_object_masks(cv_image, merge_objects = True).squeeze()
            # TODO(RN) is mask right?
            baseline = self.inpainting_model.inpaint(cv_image, mask)

        else:
            print(f"ERROR: unexpected baseline '{baseline_type}'!")
            exit(-1)

        return PIL.Image.fromarray(baseline)

    def _get_object_masks(self,
        image: Union[np.ndarray, PIL.Image.Image],
        merge_objects: bool,
        return_labels: bool = False
    ) -> np.ndarray:
        """
        Run the instance segmentation model on the given OpenCV/PIL image, and return
        the computed object mask(s) and optionally the class labels for each object.

        Args:
            image:          An image in the OpenCV or PIL format
            merge_objects:  If set, then the object masks will be merged together
            return_labels:  If set, then the class labels will be returned for each object
        
        Returns:
            The binary masks with the same shape as the image, and the class labels 
            if 'return_labels' is set.
        """
        # The instance segmentation model expects OpenCV images
        if isinstance(image, PIL.Image.Image):
            image = utils.pil_to_opencv_image(image)

        if return_labels:
            masks, labels = self.instance_segmentation_model.extract_segmentation(image, return_labels=True)
        else:
            masks = self.instance_segmentation_model.extract_segmentation(image, return_labels=False)

        if merge_objects:
            masks = masks.max(axis = 0, keepdims = True)

        if self.args.dilation > 0:
            for i in range(len(masks)):
                masks[i] = utils.dilate_mask(masks[i], n_iters=self.args.dilation)

        if return_labels:
            return masks, labels
        else:
            return masks

    def get_classifier_prediction(self, image: PIL.Image.Image) -> Tuple[int, float]:
        """
        Return the predicted label and its probability using the classifier.
        """
        preds = self.classifier.predict(image).squeeze()
        label = preds.argmax().item()
        prob = softmax(preds.detach().cpu().numpy()).max().item()

        return label, prob

def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    # Add with default arguments such as --show_plot and --output_dir
    parser = utils.create_default_argparser(output_dir = "outputs/integrated_gradients")
    
    IntegratedGradients.add_argparse_args(parser)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
    