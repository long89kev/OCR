
# Video Keyframe Analysis System

This project is a comprehensive system for analyzing video keyframes using object detection and optical character recognition (OCR). It processes video keyframes, detects objects, extracts text, and provides a user-friendly interface for searching and viewing results.

## Features

- Object detection using YOLOv5
- Optical Character Recognition (OCR) for text extraction
- Streamlit-based web interface for searching and viewing results
- Support for multiple video processing
- CSV output for object detection and OCR results

## Main Components

1. Object Detection (`obj_dectec.py`)
2. Web Interface (`app.py`)

## Installation

1. Clone the repository
2. Install the required dependencies:
3. Download the YOLOv5 model:


## Usage

1. Place your video keyframes in the `datasets/keyframes` directory, organized by video name.

2. Run the object detection script:

3. Run the OCR script (not provided in the snippets, but assumed to exist)

4. Launch the Streamlit app:

5. Use the web interface to search for text or objects in the processed keyframes.

## File Structure

- `obj_dectec.py`
- `orc.py`
- `app.py`
- `datasets/`
  - `clip-features/`
  - `keyframes/`
  - `videos/`
  - `map-keyframes/`
  - `object_detection_results/`
  - `ocr_results/`