import os
from os.path import join
import shutil
import numpy as np
from argparse import ArgumentParser
import cv2

from skimage.util import compare_images
import matplotlib.pyplot as plt

from models.wrappers.inpainting import InpaintingModel
from models.wrappers.segmentation import SegmentationModel


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="output/inpainting")
    parser.add_argument("--data_dir", type=str, default="../../data/places_small")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="The probability threshold for keeping an object segmentation mask")
    parser.add_argument("--n_dilation_iters", type=int, default=1)
    args = parser.parse_args()
    return args

def create_output_dir(output_dir):
    if os.path.exists(output_dir):
        cmd = input("-" * 16 + \
                    f"\nWARNING: output dir '{output_dir}' already exists.\n" + \
                     "\nType 'ok' to delete it and anything else to quit: ")
        if cmd != 'ok':
            exit()
        else:
            shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    

if __name__ == "__main__":
    args = parse_args()

    create_output_dir(args.output_dir)
    
    inpainting_model = InpaintingModel()
    segmentation_model = SegmentationModel(threshold=0.5)
    
    for file in os.listdir(args.data_dir):
        print(file)
        image = cv2.imread(join(args.data_dir, file))
        overlay = segmentation_model.draw_segmentation_overlay(image)
        # plt.imshow(overlay)
        # plt.show()

        masks, labels = segmentation_model.extract_segmentation(image, return_labels=True)
        
        for mask, label in zip(masks, labels):
            n_cols = int(np.ceil(args.n_dilation_iters)) + 1
            _, axes = plt.subplots(2, n_cols, gridspec_kw = {'wspace': 0, 'hspace': 0.1}, sharex='col')
            
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
            axes[1, 0].set_title("dilation = 0")

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))            
            for i in range(args.n_dilation_iters):
                mask = cv2.dilate(mask, kernel, iterations=1)
                axes[1, 1 + i].imshow(inpainting_model.inpaint(image, mask))
                axes[0, 1 + i].imshow(image[:, :, ::-1])
                axes[0, 1 + i].imshow(mask, alpha=0.3)
                axes[1, 1 + i].set_title(f"{i+1}. dilation")    
            
            plt.tight_layout()
            plt.savefig(join(args.output_dir, file))

            break
