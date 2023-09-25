"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path

import torch
from model_builder import TinyVGG


def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
    hyperparameters: dict = None,
):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
        },
        f=model_save_path,
    )


def load_model(model_path: str, device: torch.device = "cpu") -> torch.nn.Module:
    """Loads a model from a given path and sends it to a specified device."""

    # Check if model_path exists
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found at {model_path}.")

    checkpoint = torch.load(model_path)
    loaded_hyperparams = checkpoint["hyperparameters"]

    # Load in model from path
    loaded_model = TinyVGG(
        input_shape=3, hidden_units=loaded_hyperparams["hidden_units"], output_shape=3
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    return loaded_model.to(device)
