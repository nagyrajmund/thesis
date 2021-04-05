import os
from os.path import join
import numpy as np
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm
from models.wrappers.inpainting import InpaintingModel
from models.wrappers.segmentation import InstanceSegmentationModel
from scripts import utils


def parse_args():
    parser = utils.create_default_argparser(output_dir = "outputs/mask_dilation")

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="The probability threshold for keeping an object segmentation mask")
    parser.add_argument("--n_dilation_iters", type=int, default=7,
                        help="The max number of dilation iterations that are considered")

    args = parser.parse_args()
    
    return args

def main():
    """
    This script shows the effect of using increased (dilated) masks for inpainting
    objects from the image.

    TODO(RN): currently this only considers one object, but we might want to remove
              all of them.
    """
    inpainting_model = InpaintingModel()
    segmentation_model = InstanceSegmentationModel(threshold=0.5)
    
    progress_bar = tqdm(os.listdir(args.data_dir))
    for file in progress_bar:
        progress_bar.set_description(file)
        image = cv2.imread(join(args.data_dir, file))
        # overlay = segmentation_model.draw_segmentation_overlay(image)
        # plt.imshow(overlay)
        # plt.show()

        masks, labels = segmentation_model.extract_segmentation(image, return_labels=True)
        
        for mask, label in zip(masks, labels):
            n_cols = int(np.ceil(args.n_dilation_iters)) + 1
            n_rows = 2
            _, axes = plt.subplots(n_rows, n_cols, gridspec_kw = {'wspace': 0, 'hspace': 0.1}, figsize=utils.get_figsize(n_rows, n_cols))
            
            for _, axis in np.ndenumerate(axes):
                axis.axis('off')
            
            axes[0, 0].imshow(image[:,:,::-1])
            axes[0, 0].imshow(mask, alpha = 0.3)
            axes[0, 0].set_title("original image".format(
                segmentation_model.class_names[label])
            )

            # Create RGB Pillow Image from binary mask
            result = inpainting_model.inpaint(image, mask)
            axes[1, 0].imshow(result)
            axes[1, 0].set_title("no dilation")

            for i in range(args.n_dilation_iters):
                mask = utils.dilate_mask(mask, n_iters = 1)
                axes[1, 1 + i].imshow(inpainting_model.inpaint(image, mask))
                axes[0, 1 + i].imshow(image[:, :, ::-1])
                axes[0, 1 + i].imshow(mask, alpha=0.3)
                axes[1, 1 + i].set_title(f"{i+1}. dilation")    
            
            if args.show_plot:
                plt.show()
            else:
                plt.savefig(join(args.output_dir, file), bbox_inches='tight')

            break

if __name__ == "__main__":
    args = parse_args()
    utils.create_output_dir(args)
    utils.save_args_to_output_dir(args)    

    main()
