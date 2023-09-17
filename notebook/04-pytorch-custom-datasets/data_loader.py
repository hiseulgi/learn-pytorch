import os
import pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List

class ImageFolderCustom(Dataset):
    
    def __init__(self, target_dir: str, transform=None) -> None:
        
        # must know the datasets folder directory before write this
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))
        
        self.transform = transform
        
        self.classes, self.classes_to_idx = find_classes(target_dir)
    
    # function to load images
    def load_image(self, index:int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.path[index]
        return Image.open(image_path)
    
    # overwrite __len__() method
    def __len__(self) -> int:
        return len (self.paths)
    
    # overwrite __getitem__() method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.classes_to_idx[class_name]
        
        # transfrom if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx