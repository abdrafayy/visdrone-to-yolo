import os
import cv2
from tqdm import tqdm

# Set paths
ANNOTATION_DIR = "/home/Downloads/VisDrone2019-MOT-train/annotations"
SEQUENCE_DIR = "/home/Downloads/VisDrone2019-MOT-train/sequences"
YOLO_LABEL_DIR = "/home/Downloads/VisDrone2019-MOT-train/yolo_labels"

os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

def convert_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

# Main loop through all annotation files
for annotation_file in tqdm(os.listdir(ANNOTATION_DIR), desc="Converting annotations"):
    if not annotation_file.endswith(".txt"):
        continue

    sequence_name = annotation_file.replace(".txt", "")
    annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
    sequence_img_dir = os.path.join(SEQUENCE_DIR, sequence_name)

    if not os.path.exists(sequence_img_dir):
        print(f"[WARNING] Sequence images not found for {sequence_name}")
        continue

    # Get a sample image to extract image size
    sample_img_path = os.path.join(sequence_img_dir, "0000001.jpg")
    sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        print(f"[WARNING] Sample image not found or unreadable: {sample_img_path}")
        continue
    img_h, img_w = sample_img.shape[:2]

    # Prepare output directory for this sequence
    sequence_output_dir = os.path.join(YOLO_LABEL_DIR, sequence_name)
    os.makedirs(sequence_output_dir, exist_ok=True)

    # Dict to accumulate per-frame labels
    frame_to_lines = {}

    with open(annotation_path, "r") as ann_file:
        for line in ann_file:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            frame_id = int(parts[0])
            x, y, w, h = map(int, parts[2:6])
            score = int(parts[6])
            category_id = int(parts[7])
            truncation = int(parts[8])
            occlusion = int(parts[9])

            # Skip if not a valid bbox or low visibility
            if w <= 0 or h <= 0 or score == 0:
                continue

            # Convert to YOLO
            x_c, y_c, w_n, h_n = convert_to_yolo(x, y, w, h, img_w, img_h)
            yolo_line = f"{category_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

            frame_key = f"{frame_id:07d}"
            if frame_key not in frame_to_lines:
                frame_to_lines[frame_key] = []
            frame_to_lines[frame_key].append(yolo_line)

    # Write YOLO label files per frame
    for frame_key, lines in frame_to_lines.items():
        output_label_path = os.path.join(sequence_output_dir, f"{frame_key}.txt")
        with open(output_label_path, "w") as f:
            f.write("\n".join(lines))

print("[✅ DONE] YOLO format labels saved under:", YOLO_LABEL_DIR)
