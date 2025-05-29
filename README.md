# visdrone-to-yolo

# VisDrone to YOLO Converter 🛰️➡️📦

This repository provides tools to convert the [VisDrone2019 Multi-Object Tracking (MOT) train-set](https://github.com/VisDrone/VisDrone-Dataset) 
dataset annotations into the **YOLO format**, making it easier to train YOLO-based object detection and tracking models.

Additionally, it includes a **visualization tool** to verify the correctness of converted annotations by drawing bounding boxes on random frames.

---

## 📁 Repository Structure
```
visdrone-to-yolo/
├── convert_to_yolo.py # Main script to convert VisDrone annotations to YOLO format
├── visualize_yolo_annotations.py # Script to draw YOLO boxes on sample frames
├── requirements.txt # Python dependencies
├── example_output/ #  Visual output examples
└── README.md

---
```
## 📌 Features

- ✅ Converts VisDrone `.txt` annotation files to YOLO format.
- ✅ Normalizes bounding boxes for YOLO training.
- ✅ Generates **per-frame** YOLO `.txt` files for each image.
- ✅ Visual verification tool for bounding box accuracy.
- ✅ Ready for use with YOLOv5, v7, v8, etc.

---

## 🗂️ Dataset Layout Assumption

The scripts assume the following structure (as used in VisDrone2019-MOT):
```
VisDrone2019-MOT-train/
├── annotations/ # Contains 56 .txt annotation files
├── sequences/ # Contains 56 subfolders of frames/images
│ ├── uav0000001_00000_v/
│ │ ├── 0000001.jpg
│ │ └── ...
└── yolo_labels/ # Will be created for YOLO annotations
```


---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abdrafayy/visdrone-to-yolo.git
cd visdrone-to-yolo

pip install -r requirements.txt
```
🚀 How to Use
🔄 Convert Annotations to YOLO

Update the paths in convert_to_yolo.py if needed, then run:
```
python convert_to_yolo.py
```
YOLO-formatted .txt files will be saved under:

VisDrone2019-MOT-train/yolo_labels/

Each .txt corresponds to one image frame.
