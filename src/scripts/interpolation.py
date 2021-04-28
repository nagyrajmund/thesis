import PIL
from matplotlib import pyplot as plt
from models.wrappers.generation import ImageGenerator

if __name__ == "__main__":
    target_image_size = 256
    n_interpolation_steps = 10

    model = ImageGenerator(model_path = "../../utils/dall_e_checkpoint")
    image_a = PIL.Image.open("../../data/places365/validation_small/Places365_val_00000515.jpg")
    image_b = PIL.Image.open("../../data/places365/validation_small/Places365_val_00000542.jpg")

    interpolation = model.interpolate(
        image_a, image_b, n_interpolation_steps)

    _, axes = plt.subplots(1, n_interpolation_steps, gridspec_kw = {'wspace': 0, 'hspace': 0})

    for image, axis in zip(interpolation, axes):
        axis.imshow(image)
        axis.axis('off')

    plt.show()