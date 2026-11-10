import os
import random
import sys
import time
import warnings

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

if sys.platform == "win32":
    NAS_PATH = "Z:"
    BATCH_SIZE = 32
elif sys.platform == "linux":
    NAS_PATH = "/media/Z"
    BATCH_SIZE = 8
else:
    print(f"Unsupported platform: {sys.platform}")
    exit()

# Set parameters
SOURCE_PATH = os.path.join(NAS_PATH, "data/driver_monitoring/raw")
DESTINATION_PATH = os.path.join(NAS_PATH, "data/driver_monitoring/features_augmented")
TARGET_MODALITY = "RGB"
ORIGINAL_DIMENSIONS = (512, 512)
TARGET_DIMENSIONS = (224, 224)
WINDOW = 16
STRIDE = 8
N_STAGGERED_WINDOWS = 4

# Iterate through sequences
sequences = os.listdir(SOURCE_PATH)
sequences = np.random.permutation(sequences)
print(f"Found {len(sequences)} sequences in the dataset. \n")
for sequence in sequences:
    t = time.time_ns()
    print(f"Processing sequence: {sequence}")
    
    # Load the frames
    frames = np.load(os.path.join(SOURCE_PATH, sequence, "RGB.npy"))
    
    # Seed for reproducibility
    random.seed(sequence)
    
    # Flip the frames
    for flip in range(2):
    
        flipped_frames = frames.copy()
        if flip:
            flipped_frames = np.flip(flipped_frames, axis=2)
        
        # 4x staggered windows with random crops
        for stagger in range(N_STAGGERED_WINDOWS):
            
            # Random offsets for cropping
            x_offset = random.randint(-32, 32)
            y_offset = random.randint(-32, 32)
            
            # Calculate starting indices for cropping
            y_start = ((ORIGINAL_DIMENSIONS[1] - TARGET_DIMENSIONS[1]) // 2) + y_offset
            x_start = ((ORIGINAL_DIMENSIONS[0] - TARGET_DIMENSIONS[0]) // 2) + x_offset
            
            # Crop the frames
            cropped_frames = flipped_frames[:, y_start:y_start + TARGET_DIMENSIONS[1], x_start:x_start + TARGET_DIMENSIONS[0], :]
            
            # Check that the number of staggered windows is less than the stride
            assert N_STAGGERED_WINDOWS < STRIDE, "Number of staggered windows must be less than the stride."
            
            # Calculate the offset for the staggered windows
            offset = stagger * (STRIDE // N_STAGGERED_WINDOWS)
            
            # Create output paths
            torch_features_path = os.path.join(DESTINATION_PATH, f"{sequence}-{flip}-{x_start}-{y_start}-{offset}", "features.pt")
            numpy_features_path = os.path.join(DESTINATION_PATH, f"{sequence}-{flip}-{x_start}-{y_start}-{offset}", "features.npz")
            
            if os.path.exists(torch_features_path) and os.path.exists(numpy_features_path):
                print(f"Features already exist for sequence: {sequence}-{flip}-{x_start}-{y_start}-{offset}. Skipping...")
                continue
            
            # Trim the cropped frames
            input_frames = cropped_frames[offset :]
            
            # Compute starting indices for sliding windows
            starting_indices = list(range(0, input_frames.shape[0] - WINDOW + 1, STRIDE))
            
            # Initialise the features tensor
            features = torch.zeros((0, 1408), dtype=torch.float32, device="cuda")
            t_features = time.time_ns()

            # Process input_frames in batches
            while len(starting_indices) > 0:
                batch = []
                for i in range(min(BATCH_SIZE, len(starting_indices))):
                    start_index = starting_indices.pop(0)
                    window = input_frames[start_index:start_index + WINDOW]
                    batch.append(list(window))
        
                # Process the batch
                inputs = processor(batch, return_tensors="pt")
                pixel_values = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
                pixel_values = pixel_values.to("cuda")
                
                with torch.no_grad():
                    outputs = model(pixel_values)
                    
                features = torch.cat((features, outputs), dim=0)
                
            print(f"Extracted features shape: {features.shape}")
            print(f"Extracted features in {(time.time_ns() - t_features) / 1e9} s")
            
            # Unsqueeze the features
            features = features.unsqueeze(0)
            
            # Save the features
            os.makedirs(os.path.join(DESTINATION_PATH, f"{sequence}-{flip}-{x_start}-{y_start}-{offset}"), exist_ok=True)
            torch.save(features, torch_features_path)
            np.savez(numpy_features_path, feature=features.cpu().numpy())
                
    print(f"Saved features for sequence: {sequence} in {(time.time_ns() - t) / 1e9} s\n")