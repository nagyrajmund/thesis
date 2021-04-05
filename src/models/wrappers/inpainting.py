import numpy as np
import tensorflow as tf
from scripts.utils import create_log_manager

# Disable tensorflow deprecation warnings
with create_log_manager():
    import neuralgym as ng
    from models.backends.deepfill.inpaint_model import InpaintCAModel

class InpaintingModel:
    """
    A wrapper around the popular DeepFill-v2 generative inpainting model.

    Args:
        disable_logs: if True, this class will not print anything in order to reduce noise.
    """
    def __init__(self, disable_logs: bool = True):
        self.log_manager = create_log_manager(disable_logs)
        
        with self.log_manager:
            self.flags = ng.Config('../models/backends/deepfill/inpaint.yml')
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
                    var_value = tf.contrib.framework.load_variable("../../utils/deepfill_checkpoint/", from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output)
                
                return result[0]

