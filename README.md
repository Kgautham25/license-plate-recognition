# License Plate Detection & OCR

_Last updated: April 24, 2026_

An end-to-end deep-learning pipeline that detects vehicle license plates in an image with **YOLOv5**, then reads the plate text with **EasyOCR**.

> Final project for Deep Learning (CS 747), George Mason University.

## Overview

The pipeline has two stages:
1. **Detection** — a YOLOv5 model locates the license-plate region in each image.
2. **Recognition** — the cropped plate is passed to EasyOCR to extract the characters.

The model is trained on the public [Roboflow License Plate Recognition](https://universe.roboflow.com/) dataset (~10,000 annotated images in YOLO format).

## Features

- Trains a YOLOv5 detector on a Roboflow dataset.
- Runs OCR on detected plates with EasyOCR.
- Visualizes detections and recognized text on sample images.

## Tech stack

Python · PyTorch · Ultralytics YOLOv5 · EasyOCR · OpenCV · Roboflow · Matplotlib

## Getting started

Open `License_Plate_Detection_OCR.ipynb` in Google Colab or Jupyter (a GPU runtime is recommended for training).

### Install dependencies
```bash
pip install -r requirements.txt
```

### Roboflow API key
The dataset download needs a Roboflow API key. The notebook reads it from an environment variable, so set it first instead of pasting it into a cell:
```python
import os
os.environ["ROBOFLOW_API_KEY"] = "your-key-here"
```

### Run
Execute the notebook cells top to bottom: download the dataset, train the detector, then run detection + OCR on test images.

## Results

Sample detection/OCR outputs and a training chart are in the [`results/`](results/) folder.

## Notes

- Trained weights (`*.pt`) and the raw dataset are **not** committed — they are regenerated/downloaded by the notebook. See `.gitignore`.

## Author

Gautham Kumar Karamthot
