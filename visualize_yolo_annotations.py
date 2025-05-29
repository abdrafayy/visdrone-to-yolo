import os
import cv2
import random

# Paths (update if different)
IMAGE_ROOT = "/home/Downloads/VisDrone2019-MOT-train/sequences"
YOLO_LABEL_ROOT = "/home/Downloads/VisDrone2019-MOT-train/yolo_labels"
OUTPUT_DIR = "/home/Downloads/VisDrone2019-MOT-train/debug_vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of samples to visualize
NUM_SAMPLES = 5

# Helper function to draw YOLO bboxes
def draw_yolo_boxes(image, labels, class_names=None):
    h, w = image.shape[:2]
    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_c, y_c, box_w, box_h = map(float, parts)
        x_c *= w
        y_c *= h
        box_w *= w
        box_h *= h

        x1 = int(x_c - box_w / 2)
        y1 = int(y_c - box_h / 2)
        x2 = int(x_c + box_w / 2)
        y2 = int(y_c + box_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = str(int(class_id)) if class_names is None else class_names[int(class_id)]
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# Step 1: Randomly pick a few sequence folders
sequences = os.listdir(YOLO_LABEL_ROOT)
sequences = [seq for seq in sequences if os.path.isdir(os.path.join(YOLO_LABEL_ROOT, seq))]
selected_sequences = random.sample(sequences, min(NUM_SAMPLES, len(sequences)))

# Step 2: For each sequence, pick a random frame and draw bounding boxes
for seq in selected_sequences:
    label_folder = os.path.join(YOLO_LABEL_ROOT, seq)
    image_folder = os.path.join(IMAGE_ROOT, seq)

    label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
    if not label_files:
        continue

    selected_label_file = random.choice(label_files)
    frame_name = selected_label_file.replace(".txt", ".jpg")
    
    label_path = os.path.join(label_folder, selected_label_file)
    image_path = os.path.join(image_folder, frame_name)

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    with open(label_path, "r") as f:
        labels = f.readlines()

    image_with_boxes = draw_yolo_boxes(image, labels)
    out_path = os.path.join(OUTPUT_DIR, f"{seq}_{frame_name}")
    cv2.imwrite(out_path, image_with_boxes)
    print(f"[✓] Saved annotated image: {out_path}")

print(f"\n[✅ DONE] Sample images with bounding boxes saved in:\n{OUTPUT_DIR}")
