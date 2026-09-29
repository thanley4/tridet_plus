import os
import time
import warnings

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor


# Ignore warnings from Hugging Face libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Load the model and processor
config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-giant")
model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-giant', config=config, trust_remote_code=True)
model = model.to("cuda").eval()

# Set parameters
SOURCE_PATH = "Z:/data/driver_monitoring/raw"
DESTINATION_PATH = "Z:/data/driver_monitoring/features"
TARGET_MODALITY = "RGB"
DIMENSIONS = (224, 224)
BATCH_SIZE = 32
WINDOW = 16
STRIDE = 8



sequences = os.listdir(SOURCE_PATH)
print(f"Found {len(sequences)} sequences in the dataset. \n")

for sequence in sequences:
    
    print(f"Processing sequence: {sequence}")
    pngs = os.listdir(os.path.join(SOURCE_PATH, sequence, TARGET_MODALITY))
    print(f"Found {len(pngs)} PNG files in sequence: {sequence}")
    
    # Initialize the video array
    video = np.zeros((len(pngs), *DIMENSIONS, 3), dtype=np.uint8)
    
    # Load the pngs
    for i, png in enumerate(pngs):
        
        if ".png" not in png:
            continue
        
        png_path = os.path.join(SOURCE_PATH, sequence, TARGET_MODALITY, png)
        frame = cv2.imread(png_path)
        frame = cv2.resize(frame, DIMENSIONS)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)
        video[i] = frame
    
    # Compute starting indices for sliding windows
    starting_indices = list(range(0, video.shape[0] - WINDOW + 1, STRIDE))
    print(f"Number of windows: {len(starting_indices)}")

    # Initialize the features tensor
    features = torch.zeros((0, 1408), dtype=torch.float32, device="cuda")

    # Process video in batches
    while len(starting_indices) > 0:
        batch = []
        for i in range(min(BATCH_SIZE, len(starting_indices))):
            start_index = starting_indices.pop(0)
            window = video[start_index:start_index + WINDOW]
            batch.append(list(window))

        # Process the batch
        inputs = processor(batch, return_tensors="pt")
        pixel_values = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
        pixel_values = pixel_values.to("cuda")
        
        with torch.no_grad():
            outputs = model(pixel_values)
            
        features = torch.cat((features, outputs), dim=0)
        
    print(f"Extracted features shape: {features.shape}\n")
    
    # Save the features
    os.makedirs(os.path.join(DESTINATION_PATH, sequence), exist_ok=True)
    torch_features_path = os.path.join(DESTINATION_PATH, sequence, "features.pt")
    numpy_features_path = os.path.join(DESTINATION_PATH, sequence, "features.npy")
    torch.save(features, torch_features_path)
    np.save(numpy_features_path, features.cpu().numpy())