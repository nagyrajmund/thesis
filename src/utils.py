import os, sys
from contextlib import nullcontext
import detectron2
from typing import List

def infer_detectron2_class_names(config_file: str) -> List[str]:
    """Return a list of all known class names for the model of the given config file."""
    if config_file.startswith("COCO-InstanceSegmentation"):
        dataset = "coco_2017_train"
    elif config_file.startswith("LVISv0.5-InstanceSegmentation"):
        dataset = "lvis_v0.5_train"
    elif config_file.startswith("LVISv1-InstanceSegmentation"):
        dataset = "lvis_v1_train"
    else:
        raise Exception(f"Could not infer dataset name from config file '{config_file}'")

    return detectron2.data.MetadataCatalog.get(dataset).thing_classes


class DisablePrints:
    """Context manager to disable printing e.g. in a function call."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def create_log_manager(disable_prints: bool):
    """
    Return a context manager for hiding print() outputs if 'disable_prints' is True,
    else return a dummy context manager that does nothing. 
    """
    if disable_prints:
        return DisablePrints()
    else:
        return nullcontext()