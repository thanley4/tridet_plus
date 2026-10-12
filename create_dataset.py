import json
import os

import matplotlib.pyplot as plt
import numpy as np


def find_events(array):
    events = []
    in_event = False
    start_idx = 0

    for i in range(len(array)):
        if array[i] == 1 and not in_event:
            in_event = True
            start_idx = i
        elif array[i] == 0 and in_event:
            in_event = False
            end_idx = i - 1
            events.append((start_idx, end_idx))

    if in_event:
        events.append((start_idx, len(array) - 1))

    return events


DATASET_PATH = "Z:/data/driver_monitoring/raw"
JSON_PATH = "tridet_plus/data/driving.json"
FPS = 30
TRAIN_ASSETS = ["bridget", "grace", "james", "john", "liam", "mary"]
TEST_ASSETS = ["olivia", "patrick"]

sequences = os.listdir(DATASET_PATH)
print(f"Found {len(sequences)} sequences in the dataset.")

# Initialise the dataset dictionary
dataset = {
    "version": "driving-30fps",
    "database": {}
}

# Iterate through each sequence and process the events
for i, sequence in enumerate(sequences):

    print(f"Processing sequence: {sequence}")
    blinks = np.load(os.path.join(DATASET_PATH, sequence, "blinks.npz"))
    key = blinks.files[0]
    asset = sequence.split("_")[0]
    
    # Initialize the sequence entry in the dataset dictionary
    dataset["database"][sequence] = {}
    
    if asset in TRAIN_ASSETS:
        dataset["database"][sequence]["subset"] = "Train"
    elif asset in TEST_ASSETS:
        dataset["database"][sequence]["subset"] = "Test"
    else:
        print(f"Warning: Asset {asset} not found in TRAIN or TEST lists. Assigning to 'Unknown' subset.")
        dataset["database"][sequence]["subset"] = "Unknown"
        
    dataset["database"][sequence]["fps"] = FPS 
    dataset["database"][sequence]["duration"] = int(blinks[key].shape[0] / FPS)
    dataset["database"][sequence]["annotations"] = []
    
    # Classes: 0 = blink, 1 = fidgeting, 2 = yawning, 3 = fixation, 4 = microsleep, 5 = head_nod
    # Store the blink events in the dataset dictionary
    blinks = np.load(os.path.join(DATASET_PATH, sequence, "blinks.npz"))[key]
    blink_events = find_events(blinks)
    for start, end in blink_events:
        dataset["database"][sequence]["annotations"].append({
            "label": "blink",
            "segment": [start / FPS, end / FPS],
            "label_id": 0
        })
    
    # # Store the fidgeting events in the dataset dictionary
    # fidgeting = np.load(os.path.join(dataset_path, sequence, "fidgeting.npz"))[key]
    # fidget_events = find_events(fidgeting)
    # for start, end in fidget_events:
    #     dataset["database"][sequence]["annotations"].append({
    #         "label": "fidgeting",
    #         "segment": [start / FPS, end / FPS],
    #         "label_id": 1
    #     })
        
    # # Store the yawning events in the dataset dictionary
    # yawns = np.load(os.path.join(dataset_path, sequence, "yawns.npz"))[key]
    # yawn_events = find_events(yawns)
    # for start, end in yawn_events:
    #     dataset["database"][sequence]["annotations"].append({
    #         "label": "yawn",
    #         "segment": [start / FPS, end / FPS],
    #         "label_id": 2
    #     })
    
    # # Store the gaze fixation events in the dataset dictionary
    # gaze_fixations = np.load(os.path.join(dataset_path, sequence, "gaze_fixations.npz"))[key]
    # fixation_events = find_events(gaze_fixations)
    # for start, end in fixation_events:
    #     dataset["database"][sequence]["annotations"].append({
    #         "label": "fixation",
    #         "segment": [start / FPS, end / FPS],
    #         "label_id": 3
    #     })
    
    # # Store the microsleep events in the dataset dictionary
    # microsleeps = np.load(os.path.join(dataset_path, sequence, "microsleeps.npz"))[key]
    # microsleep_events = find_events(microsleeps)
    # for start, end in microsleep_events:
    #     dataset["database"][sequence]["annotations"].append({
    #         "label": "microsleep",
    #         "segment": [start / FPS, end / FPS],
    #         "label_id": 4
    #     })
    
    # # Store the head nod events in the dataset dictionary
    # head_nods = np.load(os.path.join(dataset_path, sequence, "head_nods.npz"))[key]
    # head_nod_events = find_events(head_nods)
    # for start, end in head_nod_events:
    #     dataset["database"][sequence]["annotations"].append({
    #         "label": "head_nod",
    #         "segment": [start / FPS, end / FPS],
    #         "label_id": 5
    #     })
    
    
# Save the dataset as a JSON file
with open(JSON_PATH, "w") as f:
    json.dump(dataset, f)