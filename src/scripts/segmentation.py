import cv2

from skimage.util import compare_images
import matplotlib.pyplot as plt

from models.wrappers.inpainting import InpaintingModel
from models.wrappers.segmentation import SegmentationModel

if __name__ == "__main__":
    inpainting_model = InpaintingModel()
    segmentation_model = SegmentationModel(threshold=0.5)
    image = cv2.imread("../../data/places_small/Places365_val_00000173.jpg")
    
    overlay = segmentation_model.draw_segmentation_overlay(image)
    plt.imshow(overlay)
    plt.show()

    masks, labels = segmentation_model.extract_segmentation(image, return_labels=True)
    
    for mask, label in zip(masks, labels):
        # Create RGB Pillow Image from binary mask

        result = inpainting_model.inpaint(image, mask)

        ax = plt.subplot(131)
        ax.imshow(image[:,:,::-1])
        ax.imshow(mask, alpha = 0.2)
        plt.title("original image and mask ({})".format(
            segmentation_model.class_names[label])
        )

        ax = plt.subplot(132)
        ax.imshow(result)
        plt.title("inpainted image")

        ax = plt.subplot(133)
        ax.imshow(compare_images(image[:,:,::-1], result, method='diff'))
        plt.title("differences between original and inpainted image")

        plt.show()