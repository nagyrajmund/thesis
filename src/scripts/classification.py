import os
from os.path import join

import PIL
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import functional as F
from torch.nn.functional import softmax

from models.wrappers.classification import SceneRecognitionModel

if __name__ == "__main__":
    model = SceneRecognitionModel()

    for file in os.listdir("../../data/places_small/"):
        file = join("../../data/places_small/", file)

        image = PIL.Image.open(file)
        C = model.dataset.n_semantic_classes
        semantic_scores = F.to_pil_image(torch.rand((3, 224, 224)))
        semantic_mask = F.to_pil_image(torch.randint(0, 1, (3, 224, 224)).float())
        plt.imshow(image)
        pred = model.predict(image, semantic_mask, semantic_scores)
        pred = softmax(pred.squeeze(), dim=0)
        class_probs, class_idxs = torch.topk(pred, k=5)
        plt.title(
            "\n".join(
                model.dataset.classes[class_idxs[i]] + f" ({100 * class_probs[i]:.2f}%)"
                for i in range(len(class_probs))))
        plt.show()

    