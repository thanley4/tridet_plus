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
    
    sequence_name = "-".join(augmented_sequence_name.split("-")[: -3])
    y_start = int(augmented_sequence_name.split("-")[-1])
    x_start = int(augmented_sequence_name.split("-")[-2])
    sequence_data = dataset["database"][augmented_sequence_name]
    
    frames = np.load(f"Z:/data/driver_monitoring/raw/{sequence_name}/RGB.npy")

    blink_frames = np.zeros(1800, dtype=np.uint8)
    
    for annotation in sequence_data["annotations"]:
        if annotation["label"] == "blink":
            start_frame = int(annotation["segment"][0] / (1/30))  # Convert seconds to frames
            end_frame = int(annotation["segment"][1] / (1/30))  # Convert seconds to frames
            blink_frames[start_frame:end_frame + 1] = 1
    
    # plt.figure(figsize=(15, 5))
    # plt.plot(blink_frames)
    # plt.show(False)
    
    cropped_frames = frames[:, y_start:y_start + 224, x_start:x_start + 224, :]
    
    for i, (frame, cropped_frame) in enumerate(zip(frames, cropped_frames)):
        
        if blink_frames[i] == 1:
            cv2.putText(frame, "Blink", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(cropped_frame, "Blink", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Full Frame", frame)
        cv2.imshow("Cropped Frame", cropped_frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()