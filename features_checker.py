import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load the driving JSON file
with open("tridet_plus/data/driving.json", "r") as f:
    dataset = json.load(f)
    
print(f"Loaded dataset with {len(dataset['database'])} sequences.")

augmented_sequence_names = list(dataset["database"].keys())
augmented_sequence_names = np.random.permutation(augmented_sequence_names)

for augmented_sequence_name in augmented_sequence_names:
    
    # Extract info
    sequence_name = "-".join(augmented_sequence_name.split("-")[: -3])
    y_start = int(augmented_sequence_name.split("-")[-1])
    x_start = int(augmented_sequence_name.split("-")[-2])
    sequence_data = dataset["database"][augmented_sequence_name]
    
    # Load the feature
    features = np.load(f"Z:/data/driver_monitoring/features_augmented/{augmented_sequence_name}/features.npz")
    feature = features["feature"][0]
    
    # Load annotations
    blink_frames = np.zeros(1800, dtype=np.uint8)
    fidgeting_frames = np.zeros(1800, dtype=np.uint8)
    yawn_frames = np.zeros(1800, dtype=np.uint8)
    gaze_fixations = np.zeros(1800, dtype=np.uint8)
    microsleep_frames = np.zeros(1800, dtype=np.uint8)
    head_nod_frames = np.zeros(1800, dtype=np.uint8)
    for annotation in sequence_data["annotations"]:
        if annotation["label"] == "blink":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            blink_frames[start_frame:end_frame + 1] = 1
        if annotation["label"] == "fidgeting":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            fidgeting_frames[start_frame:end_frame + 1] = 1
        if annotation["label"] == "yawn":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            yawn_frames[start_frame:end_frame + 1] = 1
        if annotation["label"] == "fixation":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            gaze_fixations[start_frame:end_frame + 1] = 1
        if annotation["label"] == "microsleep":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            microsleep_frames[start_frame:end_frame + 1] = 1
        if annotation["label"] == "head_nod":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            head_nod_frames[start_frame:end_frame + 1] = 1
    
    # Create 2 subplots: one for the feature and one for the blink events one on top of other
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(feature)
    plt.title(f"Feature for {augmented_sequence_name}")
    plt.xlabel("Frame")
    plt.ylabel("Feature Value")

    plt.subplot(2, 1, 2)
    plt.plot(blink_frames, label="Blink")
    plt.plot(fidgeting_frames, label="Fidgeting")
    plt.plot(yawn_frames, label="Yawn")
    # plt.plot(gaze_fixations, label="Gaze Fixation")
    plt.plot(microsleep_frames, label="Microsleep")
    plt.plot(head_nod_frames, label="Head Nod")
    plt.title("Blink Events")
    plt.xlabel("Frame")
    plt.ylabel("Blink")
    plt.legend()

    plt.show()