# Fruit Detection using YOLOv8

An AI-based object detection system that detects and classifies fruit images (apple, orange, banana, pineapple, peach, guava) using **YOLOv8 (Ultralytics)**.

This project consists of two parts:
1. **Training** — done in Google Colab (`training.ipynb`) using a dataset from Roboflow/Kaggle.
2. **Inference** — a Python script (`inference.py`) that detects objects in an image and displays the result in a pop-up window.

---

## Project Structure

```
.
├── training.ipynb        # Notebook for model training (Google Colab)
├── inference.py           # Inference script (detection + pop-up window)
├── best.pt                 # Trained model weights
├── apple.jpeg              # Sample test image
└── README.md
```

## Dataset

Dataset used: [Fruits by YOLO — Fruits Detection (Kaggle)](https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection)

In the training notebook, the dataset is pulled directly from **Roboflow** (workspace `fruitsdetection`, project `fruits-by-yolo`) in YOLOv8 format.

Classes detected:
- Apple
- Banana
- Grapes
- Kiwi
- Mango
- Orange
- Pineapple
- Sugerapple
- Watermelon

---

## Requirements

- Python 3.9+
- pip

### Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model implementation (training & inference) |
| `opencv-python` | Displays the pop-up window with detection results |
| `roboflow` | (training only) downloads the dataset from Roboflow |

---

## Getting Started

### 1. Clone the repo & create a virtual environment (optional but recommended)

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

For **inference only** (no need to retrain):

```bash
pip install ultralytics opencv-python
```

For **training** (run in Google Colab, not locally), the extra dependency is already installed via the notebook cells:

```bash
pip install ultralytics roboflow
```

### 3. Prepare the model

Make sure the `best.pt` file (trained model) is in the project root. If you don't have it yet, run `training.ipynb` on Google Colab first to generate it (see the [Training](#training) section).

### 4. Run Inference

Edit the image path you want to detect in the configuration section of `inference.py`:

```python
MODEL_PATH = "best.pt"
IMAGE_PATH = "apple.jpeg"
```

Then run:

```bash
python inference.py
```

A pop-up window will appear showing the image with bounding boxes, class labels, and confidence scores. Press any key to close the window.

---

## Training

Training is done on **Google Colab** using `training.ipynb`, following this workflow:

1. Install `ultralytics` and `roboflow`.
2. Download the dataset from Roboflow (YOLOv8 format) to `/content/dataset_clean`.
3. Load the pretrained base model `yolov8s.pt` and fine-tune it:
   - `epochs=100`, `imgsz=640`, `batch=16`
   - Augmentation: `mosaic=1.0`, `copy_paste=0.3`, `scale=0.5`
   - `patience=20` (early stopping)
4. Evaluate the model (`mAP50`, `mAP50-95`, precision, recall, per-class AP) and visualize the confusion matrix, training curves, and PR curve.
5. Download the resulting `best.pt` for use in the inference stage.

To retrain the model, open `training.ipynb` in Google Colab, provide your Roboflow API Key, and run all cells in order.

---

## Notes

- The image path for inference is set directly inside the script (no terminal input), as required by the task.
- The model and libraries used are fully based on the official YOLO (Ultralytics) distribution.
