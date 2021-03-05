import sys

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from matplotlib import pyplot as plt
from skimage.util import compare_images

from src.generative_inpainting.inpaint_model import InpaintCAModel
from src.utils import create_log_manager

class InpaintingModel:
    """
    A wrapper around the popular DeepFill-v2 generative inpainting model.

    Args:
        disable_tf_logs: if True, this class will not produce any tensorflow logs.
    """
    def __init__(self, disable_tf_logs: bool = True):
        self.log_manager = create_log_manager(disable_tf_logs)
        
        with self.log_manager:
            self.flags = ng.Config('generative_inpainting/inpaint.yml')
            self.model = InpaintCAModel()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint a region of the image using the given mask.

        Args:
            image:      An OpenCV image of shape (H, W, C)
            mask:       A binary OpenCV image of shape (H, W, C)

        Returns:
            The inpainted image.
        """
        with self.log_manager:
            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)        

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            
            with tf.Session(config=sess_config) as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = self.model.build_server_graph(self.flags, input_image, reuse=tf.AUTO_REUSE)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                # load pretrained model
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable("utils/deepfill_checkpoint/", from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output)
                
                return result[0]

#TODO(RN) remove
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = "../data/places_small/Places365_val_00000178.jpg"

    if len(sys.argv) >= 3:
        mask_path = sys.argv[2]
    else:
        mask_path = "../../_thesis_old/u2-net-segmentation/test_data/u2netp_results/res_0177.png"

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