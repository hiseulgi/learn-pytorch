"""
Contains functionality for get dataset from cloud and checking if it exists.
"""

import os
import zipfile
from pathlib import Path

import requests


def get_data(
    data_path: str,
):
    """
    Check data on folder and get it.

    Takes in a data directory path and check if it exists. If it doesn't exist, download it and prepare it.

    Args:
      data_path: Path to data directory.

    Example usage:
        get_data(data_path=path/to/data_dir)
    """

    # Setup path to data folder
    # data_path = Path("data/")
    data_path = Path(data_path)
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
        return
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(data_path / "pizza_steak_sushi.zip")
