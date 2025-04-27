import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore

def predict_image(model, class_names, img_size):
    print("\nWould you like to classify a new image? (yes/no)")
    user_response = input("Enter 'yes' to continue: ").strip().lower()

    if user_response != "yes":
        print("Exiting...")
        return

    img_path = input("Enter the full path to the aircraft image (.jpg or .png): ").strip()

    try:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Ensure model is in eval mode and on the correct device
        model.eval()
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_class = class_names[predicted_idx]

        print("---------------------------------")
        print(f"\nThis aircraft is a/an: {predicted_class}")
        print("----------------------------------")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the file path is correct and the image is valid.")
