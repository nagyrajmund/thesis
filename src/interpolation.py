from os.path import join
import numpy as np
import torch
import dall_e
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import PIL
from matplotlib import pyplot as plt
from typing import List

class ImageGenerator:
    """
    A pretrained discrete VAE from OpenAI's DALL-E.

    Args:
        model_path:         The path to the downloaded encoder/decoder checkpoints
        device:             The device to construct the model on (e.g. 'cpu' or 'cuda:0') 
        target_image_size:  The width/height of the images (e.g. 256 if the images are 256x256)
        
    Heavily based on the official DALL-E code: 
        https://github.com/openai/DALL-E
    """
    def __init__(self, 
        model_path: str = "../utils/dall_e_checkpoint", 
        device: str = 'cpu', 
        target_image_size: int = 256
    ):
        self.device = torch.device(device)
        self.target_image_size = target_image_size

        self.encoder = dall_e.load_model(join(model_path, "encoder.pkl"), self.device)
        self.decoder = dall_e.load_model(join(model_path, "decoder.pkl"), self.device)

    def preprocess(self, image: PIL.Image) -> torch.tensor:
        """Preprocess the given image and return it as a tensor."""
        s = min(image.size)
    
        if s < self.target_image_size:
            raise ValueError(f'min dim for image {s} < {self.target_image_size}')
            
        r = self.target_image_size / s
        s = (round(r * image.size[1]), round(r * image.size[0]))
        image = TF.resize(image, s, interpolation=PIL.Image.LANCZOS)
        image = TF.center_crop(image, output_size=2 * [self.target_image_size])
        image = torch.unsqueeze(T.ToTensor()(image), 0)
        
        return dall_e.map_pixels(image)

    def encode(self, image: PIL.Image) -> torch.tensor:
        """Return the latent code of the given image."""
        x = self.preprocess(image, self.target_image_size)
        z_logits = self.encoder(x)
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()

        return z

    def decode(self, z: torch.tensor) -> PIL.Image:
        """Decode the given latent code into an image."""
        x_stats = self.decoder(z).float()
        x_rec = dall_e.unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

        return x_rec

    def reconstruct(self, image: PIL.Image):
        """Return the reconstruction of the given image."""
        return self.decode(self.encode(image, self.target_image_size))

    def interpolate(self, 
        image_a           : PIL.Image.Image, 
        image_b           : PIL.Image.Image, 
        n_steps           : int = 5,
        add_endpoints     : bool = False
    ) -> List[PIL.Image.Image]:
        """
        Return a latent linear interpolation between the two images as a list of PIL images.
        """
        z_a = self.encode(image_a, self.target_image_size)
        z_b = self.encode(image_b, self.target_image_size)

        alphas = np.linspace(0, 1, n_steps)
        interpolation_path = torch.stack([(1-a)*z_a + a*z_b for a in alphas])
        
        interpolation = [self.decode(z) for z in interpolation_path]

        if add_endpoints:
            interpolation = [image_a] + interpolation + [image_b]

        return interpolation

if __name__ == "__main__":
    target_image_size = 256
    n_interpolation_steps = 10

    model = ImageGenerator(model_path = "../utils/dall_e_checkpoint")
    image_a = PIL.Image.open("../data/places_small/Places365_val_00000173.jpg")
    image_b = PIL.Image.open("../data/places_small/Places365_val_00000199.jpg")

    interpolation = model.interpolate(
        image_a, image_b, target_image_size, n_interpolation_steps)

    _, axes = plt.subplots(1, n_interpolation_steps, gridspec_kw = {'wspace': 0, 'hspace': 0})

    for image, axis in zip(interpolation, axes):
        axis.imshow(image)
        axis.axis('off')

    plt.show()