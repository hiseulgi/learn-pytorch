import argparse
from typing import List

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from utils import load_model


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cpu",
):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    target_image = transforms.ToPILImage()(target_image)
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    if class_names:
        result = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        result = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"

    print(result)


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(
        description="TinyVGG predicting script for food image classification."
    )
    parser.add_argument("--model_path", type=str, help="Path to model to load.")
    parser.add_argument("--image_path", type=str, help="Path to image to predict on.")

    args = parser.parse_args()

    MODEL_PATH = args.model_path
    IMAGE_PATH = args.image_path

    # check device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(MODEL_PATH, device)

    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )

    predict_image(
        model,
        IMAGE_PATH,
        class_names=["pizza", "steak", "sushi"],
        transform=data_transform,
        device=device,
    )


if __name__ == "__main__":
    main()
