from datetime import datetime
import math
from models.wrappers.segmentation import SemanticSegmentationModel
import cv2
from PIL.Image import blend
from argparse import Namespace
from os.path import join
import os
from typing import List, Sequence, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib
from matplotlib.gridspec import GridSpec
import numpy as np
from models.wrappers.classification import SceneRecognitionModel
from models.wrappers.generation import ImageGenerator
import PIL
import torch
from tqdm import tqdm
from scripts import utils
from pprint import pprint
from scipy.special import softmax
from torchvision.transforms.functional import center_crop, to_tensor, to_pil_image
from matplotlib import ticker
import sklearn
import scipy
# -----------------------------------------------------------------------------------

def main():
    """
    This script calculates the integrated gradients heatmaps for all images
    in the given dataset.
    """
    with open("TODO.md") as greetings:
        print("-"*80, greetings.read(), "-"*80, sep="\n\n")

    args = parse_args()
    if args.output_dir is None:
        args.output_dir = autogenerate_output_dir(args)

    utils.maybe_disable_tqdm(args)
    utils.create_output_dir(args, subdirs=["heatmaps", "curves"])
    utils.save_args_to_output_dir(args)
    
    insertion_curves = []
    insertion_auc_values = []
    deletion_curves = []
    deletion_auc_values  = []
    predicted_labels = []
    prediction_confidences = []

    explainer = IntegratedGradients(args)

    files = sorted(os.listdir(args.segmentation_dir))[:args.n_image_limit]
    progress_bar = tqdm(files)
    for idx, filename in enumerate(progress_bar):
        # Update the progress bar
        update_progress_bar(progress_bar, filename, insertion_auc_values, deletion_auc_values)

        # Open the input image    
        image = PIL.Image.open(join(args.data_dir, filename)).convert("RGB")

        # Create the baseline image and the interpolation    
        baseline = explainer.get_baseline(image, filename)
        interpolation = explainer.create_interpolation(baseline, image)

        # Select the label to condition the explanation on
        target_label, target_prob = explainer.get_classifier_prediction(image, args.target_label)
        predicted_labels.append(target_label)
        prediction_confidences.append(target_prob)

        # Compute the attribution heatmap
        attributions = explainer.compute_attributions(interpolation, target_label)
        
        # Visualize the heatmaps
        explainer.plot_heatmaps(idx, interpolation, attributions, target_label, target_prob, filename)

        # Compute the insertion/deletion curves
        insertion_curve, deletion_curve = explainer.insertion_deletion_curves(image, attributions, target_label)
        insertion_curves.append(insertion_curve)
        deletion_curves.append(deletion_curve)

        # Visualize the insertion/deletion curves
        explainer.plot_curves(insertion_curve, deletion_curve, target_label, target_prob, filename)

        # Store the AUC of the curves
        insertion_auc_values.append(auc(insertion_curve))
        deletion_auc_values.append(auc(deletion_curve))

    # Save the curves and their AUC to the output folder
    save_results(args, insertion_curves, insertion_auc_values, deletion_curves, deletion_auc_values, predicted_labels, prediction_confidences)

# -----------------------------------------------------------------------------------

class IntegratedGradients:
    """
    A flexible class for creating specific types of Integrated Gradients explanations.

    NOTE: the keyword args below are sometimes optional, depending on the configuration 
    in 'args'. For example, the generative model is only needed with latent interpolation.
    
    Args:
        args:       A Namespace with the parameters from 'self.add_argparse_args()'
    """
    def __init__(self,
        args: Namespace
    ):
        if args.use_segmentation_branch:
            semantic_segmentation_model = SemanticSegmentationModel(device=args.device)
            architecture = "RGB and Semantic"
        else:
            semantic_segmentation_model = None
            architecture = "RGB only"
        
        self.classifier = SceneRecognitionModel(
           architecture = architecture,
           device = args.device,
           do_ten_crops = False,
           segmentation_model = semantic_segmentation_model 
        )
        
        if "latent" in args.interpolation:
            self.generative_model = ImageGenerator(device=args.device)
        
        self.args = args
    
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--use_segmentation_branch", action="store_true")

        utils.add_choices("--interpolation", parser=parser,  choices=["linear-input", "linear-latent"])

        utils.add_choices("--baseline",      parser=parser,  choices=["black", "inpainted", "random", "white"])

        utils.add_choices("--deletion_type", parser=parser,  choices=["blur", "black", "grey", "white"],
                          help="The pixel value to use when replacing pixels in the deletion metric.")
        
        utils.add_choices("--heatmap_type",  parser=parser,  choices=["heatmap"])#"green-red", "product", ])
        
        utils.add_choices("--plot_type",     parser=parser,  choices=["single", "single+interpolation", "grid"])

        parser.add_argument("--grid_size", type=int, default=5,
            help="The number of rows/columns in the grid plot, if 'visualization' is set to 'grid'.")

        parser.add_argument("--n_steps", type=int, default=30,
            help="The number of steps in the integral approximation.")
                # "If 'object_centric' is set, then this is the number of steps " + \
                # "between each inpainting.")
        
        parser.add_argument("--target_label", type=int, default=None,
            help="The target label's index for IG. By default it is the" + \
                 "top-1 prediction of the classifier on the original image.")

        # TODO(RN): dilation is performed at dataset creation time                      
        # parser.add_argument("--dilation", type=int, default=7,
        #     help="If given, and the baseline is inpainted, then the " + \
        #          "binary mask of the instance segmentation network " + \
        #                         "will be diluted by this number of iterations.")

        parser.add_argument("--device", type=str, default="cpu",
            help="The name of the device to use (e.g. 'cpu' or 'cuda').")

        parser.add_argument("--heatmap_plot_top_pct", type=float, default=100,
            help="This number sets the threshold for showing only the top n percent attributions.")
        
        parser.add_argument("--n_insertion_bins", type=int, default=100,
            help="The number of discrete bins in the insertion curve.")
        
        parser.add_argument("--n_deletion_bins", type=int, default=200,
            help="The number of discrete bins in the deletion curve.")

        parser.add_argument("--use_unsigned_attributions", action="store_true",
            help="If set, then the absolute values of the attributions will be considered" +\
                 ", otherwise the negative attributions will be inserted/removed last")
        # parser.add_argument("--object_centric", action="store_true",
        #                help="This flag enables the object centric version of the algorithm, " + \
        #                     "where we linearly interpolate between sequentially inpainted "   + \
        #                     "versions of the original image, i.e. objects are removed one-by-one.")

        return parser

    def create_interpolation(self, 
        baseline: PIL.Image.Image, 
        image: PIL.Image.Image
    ) -> List[PIL.Image.Image]:
        """
        Return the interpolation from the baseline to the original image.
        """
        # TODO(RN): object centric is not yet implemented
        # if self.args.object_centric:
        #     return self._create_interpolation_object_centric(baseline, image)
        # else:
        return self._create_interpolation(baseline, image)

    def insertion_deletion_curves(self,
        original_image_pil: PIL.Image.Image, 
        attributions: torch.Tensor,
        target_label: int
    ) -> Tuple[List[float], List[float]]:
        attributions.squeeze_()
        original_image_pil = center_crop(original_image_pil, self.classifier.dataset.output_size)
        original_image_tensor = to_tensor(np.array(original_image_pil))
        
        # Get the ordered indices of the flattened image
        ordered_heatmap_idxs = np.argsort(attributions.flatten())
        
        # Convert them to 2D (x,y) indices
        ordered_heatmap_idxs = np.unravel_index(ordered_heatmap_idxs, shape=attributions.shape)
        # Stack them into the shape (n_pixels, 2)
        ordered_heatmap_idxs = np.dstack(ordered_heatmap_idxs)[0]
        # Change the sorting order to decreasing
        ordered_heatmap_idxs = np.flip(ordered_heatmap_idxs, axis=0)

        insertion_curve = self.compute_metric_curve("insertion", original_image_tensor, attributions, ordered_heatmap_idxs, target_label)
        deletion_curve = self.compute_metric_curve("deletion", original_image_tensor, attributions, ordered_heatmap_idxs, target_label)
              
        return insertion_curve, deletion_curve
    
    def compute_metric_curve(self,
        metric_name: str,
        original_image_tensor,
        attributions,
        ordered_heatmap_idxs,
        target_label
    ) -> Tuple[List[PIL.Image.Image], List[float]]:
        if metric_name == "insertion":
            starting_input = utils.blur_image(original_image_tensor)
            final_input = original_image_tensor
            n_bins = self.args.n_insertion_bins

        elif metric_name == "deletion":
            n_bins = self.args.n_deletion_bins
            starting_input = original_image_tensor

            if self.args.deletion_type == "blur":
                final_input = utils.blur_image(original_image_tensor)
            elif self.args.deletion_type == "black":
                final_input = torch.zeros_like(original_image_tensor)
            elif self.args.deletion_type == "white":
                final_input = torch.ones_like(original_image_tensor)
            elif self.args.deletion_type == "grey":
                final_input = 0.5 * torch.ones_like(original_image_tensor)
            else:
                raise ValueError(f"Unknown deletion type '{self.args.deletion_type}!")
    
        else:
            raise ValueError(f"Unknown metric '{metric_name}'!")
        
        curr_image = starting_input.clone()
        get_curr_prob = lambda: self.get_class_probability(to_pil_image(curr_image), target_label)
        
        metric_curve = [get_curr_prob()]

        n_pixels = len(attributions.flatten())

        bin_width = n_pixels // n_bins
                
        progress_bar = tqdm(
            range(0, n_pixels - bin_width, bin_width),
            desc=f"Computing {metric_name} score", 
            leave=False, total=n_bins
        )

        for i in progress_bar:
            selected_pixels = ordered_heatmap_idxs[i:i + bin_width]   

            # Restore or delete the most important remaining pixels
            for x, y in selected_pixels:
                curr_image[:, x, y] = final_input[:, x, y]

            metric_curve.append(get_curr_prob())

        return metric_curve


    def plot_heatmaps(self, image_idx, interpolation_images, attributions, label, prob, file):
        """
        Plot the attribution results as configured in 'args'.
        """
        # TODO(RN) minor refactor: its not necessary to do the entire val_transforms 
        #          it would be enough to crop the image, but we need the to_img function below
        #          for the gradients.
        if self.args.plot_type == "single":
            # Create new figure
            plt.close()
            plt.figure(constrained_layout=True)
            # Plot just one image and save it afterwards
            attribution_axis = plt.gca()
            skip_saving = False

        elif self.args.plot_type == "grid":
            # the Axes are stored in a global variable because we want to keep them between function calls
            global g_plot_axes
            
            # Create a new figure for the first image and also when the current
            # figure is filled completely
            if image_idx % self.args.grid_size ** 2 == 0:
                plt.close()
                _, g_plot_axes = plt.subplots(
                    self.args.grid_size, self.args.grid_size, 
                    constrained_layout=True, 
                    figsize=(self.args.grid_size**2, self.args.grid_size**2))
                
            # Find the row and column for the current image
            attribution_axis = g_plot_axes[image_idx // self.args.grid_size, image_idx % self.args.grid_size]
            
            skip_saving = False if (image_idx+1) % self.args.grid_size**2 == 0 else True
        
        elif self.args.plot_type == "single+interpolation":
            # Create two-row figure of the attribution overlay and the interpolation
            plt.close()
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
            plt.savefig(join(self.args.output_dir, "heatmaps", file), bbox_inches="tight", transparent=True, dpi=600)

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
            if self.args.heatmap_plot_top_pct is not None:
                threshold = np.percentile(attributions, 100 - self.args.heatmap_plot_top_pct)
                zero = torch.FloatTensor([0])
                attributions = torch.where(attributions < threshold, zero, attributions)
    
            heatmap = cv2.applyColorMap(np.uint8(255 * attributions), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255
            
            heatmap_overlay = attributions * heatmap + (1 - attributions) * original_image * 0.5
            axis.imshow(heatmap_overlay)

            
        else:
            print(f"ERROR: unexpected figure type: {self.args.plot_type}")
            exit(-1)


    def plot_curves(self,
        insertion_curve: np.ndarray, 
        deletion_curve: np.ndarray, 
        target_label: int, 
        original_prediction_prob: float,
        filename: str
    ):
        plt.close()
        fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=600, constrained_layout=False, sharey=True)
        results = [insertion_curve, deletion_curve]
        line_labels = ["insertion", "deletion"]
        x_labels = ["ratio of inserted pixels", "ratio of deleted pixels"]
        y_labels = ["prediction confidence", ""]
        colors = ["blue", "red"]

        x_values = [
            np.arange(self.args.n_insertion_bins + 1),
            np.arange(self.args.n_deletion_bins + 1)
        ]
        # Format the axes as percentages
        x_tick_formatters = [
            ticker.PercentFormatter(xmax=self.args.n_insertion_bins, decimals=False),
            ticker.PercentFormatter(xmax=self.args.n_deletion_bins, decimals=False)
        ]
        y_tick_formatter = ticker.PercentFormatter(xmax=1.0, decimals=False)        

        for i in [0,1]:
            axes[i].plot(x_values[i], results[i], label=line_labels[i], color=colors[i], linewidth=0.8)
            axes[i].fill_between(x_values[i], 0, results[i], color=colors[i], alpha=0.3)
            axes[i].legend()
            axes[i].set_ylim(bottom=0, top=1)
            axes[i].xaxis.set_major_formatter(x_tick_formatters[i])
            axes[i].yaxis.set_major_formatter(y_tick_formatter)
            axes[i].set_xlabel(x_labels[i])
            axes[i].set_ylabel(y_labels[i])

        fig.subplots_adjust(wspace=0.05)
        plt.suptitle(f"Original prediction: {self.classifier.dataset.class_names[target_label]} ({original_prediction_prob*100:.0f}%)" )

        if self.args.show_plot:
            plt.show()
        else:
            plt.savefig(join(self.args.output_dir, "curves", filename))

    def _create_interpolation(self,
        baseline: PIL.Image.Image, 
        image: PIL.Image.Image
    ) -> List[PIL.Image.Image]:
        """
        Create a lits of images containing a linear or a latent interpolation
        between the baseline and the original image.
        """
        if self.args.interpolation == "linear-input":
            interpolation_np = [
                baseline * (1-a) + image * a 
                for a in np.linspace(0, 1, num=self.args.n_steps, endpoint=True)
            ]

            interpolation_pil = [PIL.Image.fromarray(np.uint8(img)) for img in interpolation_np]
            return interpolation_pil

        elif self.args.interpolation == "linear-latent":
            interpolation = self.generative_model.interpolate(
                baseline, image, 
                n_steps = self.args.n_steps, 
                add_original_images=False
            )

            return interpolation

        else:
            raise ValueError(f"Unexpected interpolation type: {self.args.interpolation}.")

        
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

        if self.args.use_unsigned_attributions:
            attributions.abs_()

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
        
        
        
        if self.args.use_segmentation_branch:
            semantic_mask, semantic_scores = self.classifier.get_segmentation(image)
            semantic_mask = semantic_mask.to(self.args.device)
            semantic_scores = semantic_scores.to(self.args.device)
        else:
            semantic_mask = semantic_scores = None

        image_tensor = self.classifier.preprocess_img(image).to(self.args.device)
       
        # Get the classifier's prediction while tracking the gradients
        prediction = self.classifier.predict_from_tensors(
            image_tensor, semantic_mask, semantic_scores, track_image_gradients=True).squeeze_()

        # Compute and return the gradients of the image tensor
        self.classifier.model.zero_grad()
        prediction[target_label].backward()
        gradients = image_tensor.grad.detach().squeeze().cpu()

        return gradients

    def get_baseline(self, image: PIL.Image.Image, filename: str) -> PIL.Image.Image:
        """
        Return the IG baseline corresponding to the given image file.
        """
        baseline_type = self.args.baseline

        if baseline_type == "black":
            baseline = np.zeros_like(np.asarray(image), dtype=np.uint8)
        
        elif baseline_type == "white":
            baseline = 255 * np.ones_like(np.asarray(image), dtype=np.uint8)

        elif baseline_type == "random":
            shape = np.asarray(image).shape
            # Fill with Gaussian noise
            baseline = np.random.randn(*shape)
            # Convert back to PIL's uint
            baseline = np.uint8(baseline * 255)

        elif baseline_type == "inpainted":
            baseline = PIL.Image.open(join(self.args.inpainting_dir, filename)).convert("RGB")
            # No need to convert it to anything else so we can return
            return baseline
        else:
            raise ValueError(f"Unexpected baseline '{baseline_type}'.")


        return PIL.Image.fromarray(baseline)

    def get_classifier_prediction(self, 
        image: PIL.Image.Image,
        label: int = None
    ) -> Tuple[int, float]:
        """
        Return the predicted label and its probability using the classifier.
        """
        preds = self.classifier.predict(image).squeeze()
        if label is None:
            label = preds.argmax().item()
            prob = softmax(preds.detach().cpu().numpy()).max().item()
        else:
            prob = softmax(preds.detach().cpu().numpy())[label].item()

        return label, prob
    
    def get_class_probability(self,
        image: PIL.Image.Image, 
        target_label: int = None
    ) -> float:
        """
        Return the probability of the 'target_label' class, or of the top-1 prediction,
        if 'target_label' is None.
        """
        pred = self.classifier.predict(image).squeeze()
        prob = softmax(pred.detach().cpu().numpy())[target_label].item()
        
        return prob

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
        # TODO(RN): object centric is not yet implemented
        raise NotImplementedError()
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
        # TODO(RN): instance segmentation is now run before the experiment
        raise NotImplementedError()
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


def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    # Add with default arguments such as --show_plot and --output_dir
    parser = utils.create_default_argparser()
    parser.add_argument("--segmentation_dir", type=str, default="../../data/thesis/places365/segmentations",
                        help="The folder with the precomputed binary object segmentation maps.")
    parser.add_argument("--inpainting_dir", type=str, default="../../data/thesis/places365/inpaintings",
                        help="The folder with the inpainted (baseline) images.")
    IntegratedGradients.add_argparse_args(parser)
    
    return parser.parse_args()

def save_results(
    args: Namespace,
    insertion_curves: List[np.ndarray],
    insertion_auc_values: List[float],
    deletion_curves: List[np.ndarray], 
    deletion_auc_values: List[float],
    predicted_labels: List[int],
    prediction_confidences: List[float]
):
    """
    Save the raw numerical results of the experiment as well as a tab-separated
    summary text file containing the following information:

        - The date and time at the end of the experiment
        - The hyperparameters of the method:
            - baseline type
            - interpolation type
            - number of interpolation steps
            - number of insertion/deletion bins
            - the type of deletion
        - The number of images
        - The mean/std/min/max of the insertion/deletion AUC values.
    """
    insertion_metric = scipy.stats.describe(insertion_auc_values)
    deletion_metric = scipy.stats.describe(deletion_auc_values)

    
    min_ = lambda metric: metric.minmax[0]
    max_ = lambda metric: metric.minmax[1]
    std = lambda metric: np.sqrt(metric.variance)
    format = lambda x : "{:05.2%}".format(float(x))
    
    # Store the parameters and the results of the experiments in a list
    result_columns = [
        # Current date and time
        datetime.now().strftime("%d/%m/%Y"),
        datetime.now().strftime("%H:%M:%S"),
        
        # Parameters
        args.baseline,
        args.interpolation,
        "unsigned" if args.use_unsigned_attributions else "signed",
        args.n_steps,
        args.n_insertion_bins,
        args.n_deletion_bins,
        args.deletion_type,

        # Number of images
        len(insertion_auc_values),

        # Insertion results
        format(insertion_metric.mean),
        format(std(insertion_metric)),
        format(min_(insertion_metric)),
        format(max_(insertion_metric)),
        
        # Deletion results
        format(deletion_metric.mean),
        format(std(deletion_metric)),
        format(min_(deletion_metric)),
        format(max_(deletion_metric))
    ]
    # Combine the results into a tab-separated string (which can be pasted to google sheets)
    result_text = "\t".join(str(val) for val in result_columns)

    # Save the results to the experiment-specific folder
    with open(join(args.output_dir, "result.txt"), "w") as result_file:
        print(result_text, file=result_file)
    
    # Append the results to the central log file
    with open(join(args.output_dir, "..", "result_log.txt"), "a") as result_file:
        print(result_text, file=result_file)

    def save_array(array, filename):
        np.save(join(args.output_dir, filename), np.asarray(array))

    # Save all the numerical values (except AUC which can be calculated again easily)
    save_array(insertion_curves,       "insertion_curves")
    save_array(deletion_curves,        "deletion_curves")
    save_array(predicted_labels,       "predicted_labels")
    save_array(prediction_confidences, "prediction_confidences")

def autogenerate_output_dir(args: Namespace):
    """
    Return an automatically generated output directory name from the most
    important command-line parameters.
    """
    if args.use_unsigned_attributions:
        signed_unsigned_text = "with unsigned attributions,\n"
    else:
        signed_unsigned_text = "with signed attributions,\n"

    dir_name = f"{args.baseline} baseline, {args.interpolation} interpolation,\n" + \
               signed_unsigned_text + \
               f"{args.deletion_type} deletion, {args.n_image_limit} images, {args.n_steps} steps,\n" + \
               f"{args.n_insertion_bins} insertion bins, {args.n_deletion_bins} deletion bins"
    
    return join("outputs", dir_name)

def update_progress_bar(
    progress_bar: tqdm,
    filename: str,
    insertion_auc_values: List[float],
    deletion_auc_values: List[float]
):
    """
    Update the text of the given tqdm progress bar to reflect the current status
    of the experiment.
    """
    description =  f"[input {filename_to_id(filename)}]"
        
    if len(insertion_auc_values) > 0:
        description += f" ins: {describe_array(insertion_auc_values)},"
        description += f" del: {describe_array(deletion_auc_values)}"
    
    progress_bar.set_description(description)

def filename_to_id(fname: str) -> int:
    """
    Return the image index in the given Places365 input filename.
    """
    start_index = len("Places365_val_")
    end_index = -len(".jpg")

    return int(fname[start_index : end_index])

def describe_array(arr: np.ndarray) -> str:
    """
    Return the statistics of the given array as a "mean +- std" string.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    
    return f"{mean:05.2%}+-{std:05.2%}"

def auc(values: Sequence[float]):
    """
    Return the AUC for the given values, assuming that the x axis goes from 0 to 1.
    """
    loc = np.linspace(0, 1, num=len(values), endpoint=True)
    return sklearn.metrics.auc(loc, values)



if __name__ == "__main__":    
    main()
