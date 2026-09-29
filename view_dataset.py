import os
import json


dataset_path = "tridet_plus/data/charades.json"

dataset = json.load(open(dataset_path, "r"))

version = dataset["version"]
data = dataset["database"]

print(f"Dataset version: {version}")
print(f"Number of videos in the dataset: {len(data)}")


video_key = list(data.keys())[0]
video = data[video_key]

print(f"Video ID: {video_key}")

video_subset = video["subset"]
video_fps = video["fps"]
video_duration = video["duration"]
video_annotations = video["annotations"]

print(f"Video Subset: {video_subset}")
print(f"Video FPS: {video_fps}")
print(f"Video Duration: {video_duration}")
print(f"Video Annotations: {len(video_annotations)}")

annotation = video_annotations[0]
annotation_label = annotation["label"]
annotation_segment = annotation["segment"]
annotation_id = annotation["label_id"]

print(f"Annotation ID: {annotation_id}")
print(f"Annotation Label: {annotation_label}")
print(f"Annotation Segment: {annotation_segment}")