import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from transformers import AutoImageProcessor, BeitForSemanticSegmentation

# 1. Load the trained model and image processor
model_path = "nevernever69/dit-doclaynet-segmentation"  # Path to your saved model
image_processor = AutoImageProcessor.from_pretrained("dit-base-doclaynet/checkpoint-500", num_labels=11)
model = BeitForSemanticSegmentation.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# 2. Define a function to visualize the segmentation results
def visualize_segmentation(image, mask, alpha=0.7):
    """
    Visualize the segmentation mask overlaid on the original image.

    Args:
        image: PIL Image or path to image
        mask: numpy array of shape (H, W) with class indices
        alpha: transparency of the overlay
    """
    # Ensure image is a PIL Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Convert image to numpy array
    img_np = np.array(image)

    # Create a colormap for visualization (11 classes)
    colors = [
        [0, 0, 0],       # 0: Background - black
        [255, 0, 0],     # 1: Title - red
        [0, 255, 0],     # 2: Paragraph - green
        [0, 0, 255],     # 3: Figure - blue
        [255, 255, 0],   # 4: Table - yellow
        [255, 0, 255],   # 5: List - magenta
        [0, 255, 255],   # 6: Header - cyan
        [128, 0, 0],     # 7: Footer - dark red
        [0, 128, 0],     # 8: Page number - dark green
        [0, 0, 128],     # 9: Footnote - dark blue
        [128, 128, 0]    # 10: Caption - olive
    ]

    # Create a colormap
    cmap = ListedColormap(np.array(colors) / 255.0)

    # Create a colored mask
    colored_mask = cmap(mask)

    # Resize the mask to the image size
    h, w = img_np.shape[:2]
    resized_mask = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay the mask on the image
    overlay = img_np.copy()
    overlay = (overlay * (1 - alpha) + resized_mask[:, :, :3] * 255 * alpha).astype(np.uint8)

    # Plot the original image and the overlay
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(mask, cmap=cmap)
    ax2.set_title("Segmentation Mask")
    ax2.axis("off")

    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()

    # Return the visualization for further use if needed
    return overlay

# 3. Function to run inference on a new image
def segment_document(image_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform document layout segmentation on a new image.

    Args:
        image_path: Path to the image to segment
        device: Device to run inference on ("cuda" or "cpu")

    Returns:
        PIL Image, segmentation mask
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Move model to device
    model.to(device)

    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted segmentation mask
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    # Convert to class indices
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Visualize the results
    overlay = visualize_segmentation(image, predicted_mask)

    return image, predicted_mask, overlay

# 4. Example usage
if __name__ == "__main__":
    # Replace with your test image path
    test_image_path = "image-5bd08790e1864.png"

    # Run segmentation
    image, mask, overlay = segment_document(test_image_path)

    # You can also save the results
    # overlay_pil = Image.fromarray(overlay)
    # overlay_pil.save("segmentation_result.jpg")

    # Class names for reference
    class_names = [
        "Background",
        "Title",
        "Paragraph",
        "Figure",
        "Table",
        "List",
        "Header",
        "Footer",
        "Page number",
        "Footnote",
        "Caption"
    ]

    # Print statistics about the segmentation
    unique_classes = np.unique(mask)
    print("Detected document elements:")
    for cls in unique_classes:
        pixel_count = np.sum(mask == cls)
        percentage = (pixel_count / mask.size) * 100
        print(f"  - {class_names[cls]}: {percentage:.2f}% of the document")
