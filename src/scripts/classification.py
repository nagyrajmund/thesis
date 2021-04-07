import os
from os.path import join
from typing import Tuple
import PIL
from tqdm import tqdm
from models.wrappers.classification import SceneRecognitionModel
from argparse import Namespace
from scripts import utils
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import softmax

def parse_args() -> Namespace:
    """
    Return the parsed command-line arguments for this script.
    """
    parser = utils.create_default_argparser()
    parser.add_argument("--segmentation_dir", type=str, default="../../data/places365/validation_segmentations",
                        help="The path to the folder which contanis the SASceneNet semantic annotations (see SASceneNet for details)")
    args = parser.parse_args()
    return args

def main():
    """
    This script shows the classifier's outputs for images from the validation dataset.
    """   
    file_labels = load_image_labels()

    progress_bar = tqdm(os.listdir(args.data_dir))
    for file in progress_bar:
        progress_bar.set_description(file)
        
        image, semantic_mask, semantic_scores = load_image_and_segmentation(file, file_labels)
        
        pred = classifier.predict(image, semantic_mask, semantic_scores)
        pred = softmax(pred.squeeze(), dim=0)
        class_probs, class_idxs = torch.topk(pred, k=5)
        plt.imshow(image)
        plt.title("\n".join(
            classifier.dataset.class_names[class_idxs[i]] + f" ({100 * class_probs[i]:.2f}%)"
            for i in range(len(class_probs))
        ))
        plt.show()

def load_image_labels():
    """
    Load the image labels from the data directory.
    """
    label_file = join(args.data_dir, "labels.txt")
    labels = pd.read_csv(label_file, sep=" ", index_col="file", dtype={"file": str, "label": int})

    return labels

def load_image_and_segmentation(file: str, file_labels: pd.DataFrame) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    """
    Return the given file as a PIL Image and its precomputed segmentation masks/scores as arrays.
    """
    label_idx = file_labels.loc[file].label
    label_name = classifier.dataset.class_names[label_idx]

    image = PIL.Image.open(join(args.data_dir, file))
    semantic_mask = PIL.Image.open(join(args.segmentation_dir, "noisy_annotations_RGB", "val", label_name, file.replace(".jpg", ".png")))
    semantic_scores = PIL.Image.open(join(args.segmentation_dir, "noisy_scores_RGB", "val", label_name, file.replace(".jpg", ".png")))

    return image, semantic_mask, semantic_scores


if __name__ == "__main__":
    args = parse_args()
    classifier = SceneRecognitionModel()
    main()
