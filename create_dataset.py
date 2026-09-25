import os
import numpy as np
import matplotlib.pyplot as plt


dataset_path = "Z:/data/driver_monitoring"


sequences = os.listdir(dataset_path)
print(f"Found {len(sequences)} sequences in the dataset.")

for sequence in sequences:

    blinks = np.load(os.path.join(dataset_path, sequence, "blinks.npz"))
    key = blinks.files[0]
    
    blinks = np.load(os.path.join(dataset_path, sequence, "blinks.npz"))[key]
    fidgeting = np.load(os.path.join(dataset_path, sequence, "fidgeting.npz"))[key]
    gaze_fixations = np.load(os.path.join(dataset_path, sequence, "gaze_fixations.npz"))[key]
    head_nods = np.load(os.path.join(dataset_path, sequence, "head_nods.npz"))[key]
    microsleeps = np.load(os.path.join(dataset_path, sequence, "microsleeps.npz"))[key]
    
    