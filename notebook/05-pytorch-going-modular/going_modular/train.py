"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import data_setup
import engine
import get_data
import model_builder
import torch
import utils
from torchvision import transforms


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(
        description="TinyVGG training script for food image classification."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Directory of training and testing data. (default: data/)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Training learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size (default: 32)."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Training epochs (default: 10)."
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=16,
        help="Number of hidden units in TinyVGG model (default: 16).",
    )

    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    DATA_DIR = Path(args.data_dir)

    # Check data and get it if it doesn't exist
    get_data.get_data(data_path=DATA_DIR)

    # Setup directories
    train_dir = DATA_DIR / "pizza_steak_sushi/train"
    test_dir = DATA_DIR / "pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )

    # Save the model with help from utils.py
    hyperparameters_dict = {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_units": HIDDEN_UNITS,
        "learning_rate": LEARNING_RATE,
    }

    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_going_modular_script_mode_tinyvgg_model.pth",
        hyperparameters=hyperparameters_dict,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
