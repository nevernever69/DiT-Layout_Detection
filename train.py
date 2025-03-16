import numpy as np
import cv2
from datasets import load_dataset
from transformers import AutoImageProcessor, BeitForSemanticSegmentation, TrainingArguments, Trainer
from PIL import Image
import torch
import os

# 1. Load the dataset
dataset = load_dataset("nevernever69/small-DocLayNet-v1.1")

# 2. Load image processor and model
image_processor = AutoImageProcessor.from_pretrained("microsoft/dit-base", num_labels=11)
model = BeitForSemanticSegmentation.from_pretrained("microsoft/dit-base", num_labels=11)

# 3. Define preprocessing functions
def create_segmentation_mask(example):
    """Create a segmentation mask from COCO-style annotations."""
    num_classes = 11
    H, W = 1025, 1025

    # Initialize a single-channel mask for semantic segmentation
    # For semantic segmentation, we need class indices, not one-hot encoding
    mask = np.zeros((H, W), dtype=np.int64)

    if "bboxes" in example and "category_id" in example:
        for bbox, cat in zip(example["bboxes"], example["category_id"]):
            x, y, w, h = map(int, bbox)
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)
            w = np.clip(w, 0, W - x)
            h = np.clip(h, 0, H - y)
            # Assign the class index to the mask (subtract 1 to make it 0-indexed)
            mask[y:y+h, x:x+w] = cat - 1

    # Resize mask to the target resolution (e.g., 56x56)
    target_size = (56, 56)
    resized_mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

    return resized_mask

# 4. Data collator that handles preprocessing during training
def collate_fn(examples):
    """Process a batch of examples for training."""
    # Process images
    images = []
    labels = []

    for example in examples:
        # Open and convert image
        image_path = example["image"]
        if isinstance(image_path, str) and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # If image is already a PIL Image
            image = image_path if isinstance(image_path, Image.Image) else None

        if image is None:
            raise ValueError(f"Could not open image: {image_path}")

        images.append(image)

        # Create segmentation mask
        mask = create_segmentation_mask(example)
        labels.append(mask)

    # Process images with image processor
    inputs = image_processor(images=images, return_tensors="pt")


    inputs["labels"] = torch.tensor(np.array(labels), dtype=torch.long)

    return inputs

# 5. Keep only necessary columns (we need to keep image and annotation data for mask creation)
columns_to_keep = ["image", "bboxes", "category_id"]
dataset = dataset.select_columns(columns_to_keep)

# 6. Set training arguments
training_args = TrainingArguments(
    output_dir="./dit-base-doclaynet",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_steps=1000,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=False,
)

# 7. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collate_fn,
)

# 8. Start training
trainer.train()
