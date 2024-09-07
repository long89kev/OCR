import os
import csv
import pytesseract
from PIL import Image, ImageEnhance
import pandas as pd
from multiprocessing import Pool, cpu_count
import hashlib

def process_image(args):
    keyframe_path, cache_folder = args
    image = Image.open(keyframe_path)
    
    image = preprocess_image(image)
    
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    cache_file = os.path.join(cache_folder, f"{image_hash}.txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        custom_config = r'--oem 3 --psm 6 -l vie -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image, config=custom_config)
        text = clean_ocr_text(text)
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return text.strip()

def preprocess_image(image):
    image.thumbnail((800, 600))
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    return image

def clean_ocr_text(text):
    text = ''.join(char for char in text if char.isprintable())
    text = ' '.join(text.split())
    return text

def process_keyframes(input_folder, output_folder, map_folder):
    cache_folder = os.path.join(output_folder, "cache")
    os.makedirs(cache_folder, exist_ok=True)

    all_results = []

    for video_name in os.listdir(input_folder):
        video_folder = os.path.join(input_folder, video_name)
        if os.path.isdir(video_folder):
            print(f"Processing {video_name}")
            
            keyframe_files = [f for f in os.listdir(video_folder) if f.endswith('.jpg')]
            
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(process_image, [(os.path.join(video_folder, f), cache_folder) for f in keyframe_files])
            
            for f, text in zip(keyframe_files, results):
                all_results.append({
                    'video_name': video_name,
                    'frame_idx': int(f.split('.')[0]),
                    'keyframe_file': os.path.join(video_folder, f),
                    'ocr_text': text
                })
            
    output_file = os.path.join(output_folder, "all_ocr_results.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_folder = os.path.join("datasets", "keyframes")
    output_folder = os.path.join("datasets", "ocr_results")
    map_folder = os.path.join("datasets", "map-keyframes")
    
    os.makedirs(output_folder, exist_ok=True)
    
    all_results_file = os.path.join(output_folder, "all_ocr_results.csv")
    process_keyframes(input_folder, output_folder, map_folder)
