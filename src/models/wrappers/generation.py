from os.path import join
from typing import List
from tqdm import tqdm

import PIL
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import dall_e

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
        model_path: str = "../../utils/dall_e_checkpoint", 
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
        image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
        image = TF.center_crop(image, output_size=2 * [self.target_image_size])
        image = torch.unsqueeze(T.ToTensor()(image), 0)
        
        return dall_e.map_pixels(image)

    def encode(self, image: PIL.Image) -> torch.tensor:
        """Return the latent code of the given image."""
        x = self.preprocess(image)
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
        return self.decode(self.encode(image))

    def interpolate(self, 
        image_a             : PIL.Image.Image, 
        image_b             : PIL.Image.Image, 
        n_steps             : int = 5,
        add_original_images : bool = False,
        print_progress_bar  : bool = True
    ) -> List[PIL.Image.Image]:
        """
        Return a latent linear interpolation between DALL-E's reconstruction of
        the two images as a list of PIL images. 
        """
        z_a = self.encode(image_a)
        z_b = self.encode(image_b)

        alphas = np.linspace(0, 1, n_steps)
        interpolation_path = torch.stack([(1-a)*z_a + a*z_b for a in alphas])
        
        if print_progress_bar:
            progress_bar = tqdm(interpolation_path, desc="Computing the latent interpolation", leave=False)
            interpolation = [self.decode(z) for z in progress_bar]
        else:
            interpolation = [self.decode(z) for z in interpolation_path]

        if add_original_images:
            interpolation = [image_a] + interpolation + [image_b]

        return interpolation