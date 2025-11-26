# Person Detector — MobileNetV2 (TFLite) + MediaPipe

**Professional project README**

## Overview

This repository contains an end-to-end pipeline to train and run a lightweight object detector that classifies whether an entity in an image is a *person* or *not*. The training pipeline uses MediaPipe Model Maker (object_detector) with a MobileNet-based backbone and exports the trained model to TensorFlow Lite (`.tflite`) for efficient inference. The repository also includes a simple real-time inference script (`employDetector.py`) that uses the exported TFLite model together with MediaPipe’s Tasks API and OpenCV.

This README is written in English and crafted for a professional audience — suitable to link from LinkedIn or include in a portfolio.

---

## Contents

* `ModelTraining.ipynb` — Training notebook (Colab-ready) that prepares a COCO-format dataset, retrains a MobileNet-based object detector, and exports a `.tflite` model.
* `employDetector.py` — Python script for real-time webcam inference using the exported `modelHumans2.tflite`.
* `employee/` — Example dataset directory in COCO format. Contains `train/` and `validation/` subfolders with `images/` and corresponding JSON annotations.

---

## Key features

* Retraining using **MediaPipe Model Maker** with a MobileNet backbone.
* Exports a compact **TensorFlow Lite** model suitable for edge devices.
* Real-time webcam inference using MediaPipe Tasks + OpenCV.
* Simple directory layout for COCO-style datasets and cached TFRecord usage to speed up repeated training runs.

---

## Technologies

* Python 3.8+ (recommended)
* TensorFlow 2.x
* MediaPipe Model Maker (object_detector)
* MediaPipe Tasks (Python)
* OpenCV
* NumPy

---

## Quick start — Install requirements

Below are the essential dependencies. If you plan to run the notebook in Google Colab, install them at the top of the notebook. For a local virtual environment, use pip.

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows (PowerShell)

pip install --upgrade pip
pip install tensorflow==2.*
pip install mediapipe-model-maker
pip install mediapipe
pip install opencv-python-headless numpy
```

> Tip: Use `opencv-python` instead of `opencv-python-headless` if you want GUI support for `cv2.imshow()` on your platform.

---

## Dataset structure (COCO-like)

Place your dataset in a directory called `employee/` with these two subdirectories:

```
employee/
├─ train/
│  ├─ images/
│  └─ annotations.json   # COCO-format JSON for training
└─ validation/
   ├─ images/
   └─ annotations.json   # COCO-format JSON for validation
```

In the training notebook we set:

```python
train_dataset_path = "employee/train"
validation_dataset_path = "employee/validation"
```

When using `Dataset.from_coco_folder(...)` the loader converts data to TFRecord and caches it under the `cache_dir` you supply. Reuse the same `cache_dir` across runs to avoid duplicate caches.

---

## Training (ModelTraining.ipynb)

A short summary of the training flow used by the notebook:

1. **Imports & Setup**

   ```python
   import os
   import json
   import tensorflow as tf
   assert tf.__version__.startswith('2')
   from mediapipe_model_maker import object_detector
   ```

2. **Prepare paths**

   ```python
   train_dataset_path = "employee/train"
   validation_dataset_path = "employee/validation"
   ```

3. **Create Dataset (cached TFRecord)**

   ```python
   train_data = object_detector.Dataset.from_coco_folder(
       train_dataset_path, cache_dir="/tmp/od_data/train")
   validation_data = object_detector.Dataset.from_coco_folder(
       validation_dataset_path, cache_dir="/tmp/od_data/validation")
   ```

4. **Define training options**

   ```python
   spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
   hparams = object_detector.HParams(epochs=15, export_dir='exported_model')
   options = object_detector.ObjectDetectorOptions(
       supported_model=spec,
       hparams=hparams
   )
   ```

5. **Retrain / Create model**

   ```python
   model = object_detector.ObjectDetector.create(
       train_data=train_data,
       validation_data=validation_data,
       options=options)
   ```

6. **Export** — The model is exported to the `export_dir` as a TFLite model compatible with MediaPipe Tasks.

**Note on compute time:** Retraining can take minutes to hours depending on dataset size and hardware (GPU vs CPU). Use Google Colab GPU runtime for faster experiments.

---

## Inference script (`employDetector.py`)

This script demonstrates a minimal real-time inference loop using the TFLite model produced by the notebook.

**How it works (high level):**

* Capture frames from the webcam with OpenCV.
* Create a `vision.ObjectDetector` from the exported model (TFLite) via MediaPipe Tasks.
* Run detection on each frame and draw bounding boxes and labels.

**Run**

```bash
python employDetector.py
```

Press `q` to quit the webcam window.


