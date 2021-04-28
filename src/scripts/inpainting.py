import sys

import cv2
from matplotlib import pyplot as plt
from skimage.util import compare_images

from models.wrappers.inpainting import InpaintingModel

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = "../../data/places365/validation_small/Places365_val_00000173.jpg"

    if len(sys.argv) >= 3:
        mask_path = sys.argv[2]
    else:
        mask_path = "../../../_thesis_old/u2-net-segmentation/test_data/u2netp_results/res_0177.png"

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    model = InpaintingModel()

    result = model.inpaint(image, mask)

    ax = plt.subplot(131)
    ax.imshow(image[:,:,::-1])
    ax.imshow(mask, alpha = 0.3)
    plt.title("original image and mask")

    ax = plt.subplot(132)
    ax.imshow(result)
    plt.title("inpainted image")

    ax = plt.subplot(133)
    ax.imshow(compare_images(image[:,:,::-1], result, method='diff'))
    plt.title("differences between original and inpainted image")
    
    plt.show()