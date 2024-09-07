import torch
import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pandas as pd

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(args):
    image_path, model, conf_threshold = args
    img = cv2.imread(image_path)
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()
    filtered_detections = detections[detections[:, 4] >= conf_threshold]
    labels = filtered_detections[:, -1].astype(int)
    boxes = filtered_detections[:, :4]
    confidences = filtered_detections[:, 4]
    return labels, boxes, confidences

def process_keyframes(input_folder, output_folder, conf_threshold=0.25):
    model = load_model()
    all_results = []

    for video_name in os.listdir(input_folder):
        video_folder = os.path.join(input_folder, video_name)
        if os.path.isdir(video_folder):
            print(f"Processing {video_name}")
            
            keyframe_files = [f for f in os.listdir(video_folder) if f.endswith('.jpg')]
            
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(detect_objects, [(os.path.join(video_folder, f), model, conf_threshold) for f in keyframe_files])
            
            for f, (labels, boxes, confidences) in zip(keyframe_files, results):
                for label, box, conf in zip(labels, boxes, confidences):
                    all_results.append({
                        'video_name': video_name,
                        'frame_idx': int(f.split('.')[0]),
                        'keyframe_file': os.path.join(video_folder, f),
                        'class_id': label,
                        'class_name': model.names[label],
                        'confidence': conf,
                        'x1': box[0],
                        'y1': box[1],
                        'x2': box[2],
                        'y2': box[3]
                    })
            
    output_file = os.path.join(output_folder, "all_object_detection_results.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_folder = os.path.join("datasets", "keyframes")
    output_folder = os.path.join("datasets", "object_detection_results")
    
    os.makedirs(output_folder, exist_ok=True)
    
    all_results_file = os.path.join(output_folder, "all_object_detection_results.csv")
    process_keyframes(input_folder, output_folder)
