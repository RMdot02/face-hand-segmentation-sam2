import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import replicate
import requests
from dotenv import load_dotenv
import time
import uuid

# ------------------- Load API Key -------------------
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    print("❌ REPLICATE_API_TOKEN not found in .env file!")
    exit()
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ------------------- Input Image -------------------
image_path = "sample2.webp"
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit()

# ------------------- Load & Resize Image -------------------
MAX_DIM = 1024
image_pil = Image.open(image_path)
if image_pil.mode != 'RGB':
    image_pil = image_pil.convert('RGB')

original_cv = cv2.imread(image_path)
orig_height, orig_width = original_cv.shape[:2]

# Resize if too large
if max(orig_height, orig_width) > MAX_DIM:
    new_w = int(orig_width * MAX_DIM / max(orig_height, orig_width))
    new_h = int(orig_height * MAX_DIM / max(orig_height, orig_width))
    image_pil = image_pil.resize((new_w, new_h))
    image_pil.save("resized_input.jpg")
    image_path = "resized_input.jpg"
    original_cv = cv2.imread(image_path)
    print(f"📐 Resized to: {new_w}x{new_h}")

height, width = original_cv.shape[:2]
print(f"📷 Image loaded: {width}x{height}")

# ------------------- Detection (YOLO) -------------------
print("🔍 Detecting faces...")
face_model = YOLO("yolov8n-face.pt")
face_results = face_model(image_pil, verbose=False)
face_boxes = []
for i, box in enumerate(face_results[0].boxes.xyxy.cpu().numpy()):
    conf = face_results[0].boxes.conf[i].item()
    if conf > 0.4:
        face_boxes.append(box)

print(f"📦 Faces detected: {len(face_boxes)}")

print("🖐 Detecting hands...")
hand_model = YOLO("hand_yolov8n.pt")
hand_results = hand_model(image_pil, verbose=False)
hand_boxes = []
for i, box in enumerate(hand_results[0].boxes.xyxy.cpu().numpy()):
    conf = hand_results[0].boxes.conf[i].item()
    if conf > 0.3:
        hand_boxes.append(box)

print(f"🖐 Hands detected: {len(hand_boxes)}")

def call_sam2(box, region_type, region_id):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        min_px = area * 0.1
        max_px = area * 3

        print(f"\n🧠 {region_type.title()} {region_id} | Box: [{x1}, {y1}, {x2}, {y2}], Area: {area}")

        output = replicate.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={
                "image": open(image_path, "rb"),
                "box_prompt": f"{x1},{y1},{x2},{y2}",
                "point_prompt": f"{cx},{cy}",
                "point_label": "1",
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.82,
                "points_per_side": 16,
                "use_m2m": False
            },
            timeout=200
        )

        # Get mask URL
        mask_url = None
        if isinstance(output, dict):
            mask_url = output.get("combined_mask") or output.get("individual_masks", [None])[0]
        elif hasattr(output, "url"):
            mask_url = output.url
        elif isinstance(output, list) and len(output) > 0:
            mask_url = getattr(output[0], "url", output[0])

        if not mask_url:
            print("⚠️ No valid mask URL received.")
            return None

        response = requests.get(str(mask_url), timeout=40)
        mask = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height))

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_px = np.sum(binary == 255)
        print(f"📊 Mask pixels: {mask_px}")

        if mask_px > max_px:
            print("🚫 Mask too large — trying to extract ROI...")
            roi_mask = np.zeros_like(binary)
            roi_mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
            roi_px = np.sum(roi_mask == 255)
            print(f"📏 ROI pixels: {roi_px}")

            if min_px < roi_px < max_px:
                print("✅ ROI accepted.")
                binary = roi_mask
                mask_px = roi_px
            elif roi_px > 0:
                print("⚠️ ROI below min threshold, but keeping to avoid full rejection.")
                binary = roi_mask
                mask_px = roi_px
            else:
                print("❌ ROI unusable. Rejected.")
                return None

        elif mask_px < min_px:
            print("❌ Mask too small. Rejected.")
            return None

        # Clean and return
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print(f"✅ Final mask: {mask_px} pixels")
        return binary

    except Exception as e:
        print(f"❌ SAM2 error: {e}")
        return None
# ------------------- Process Regions -------------------
face_masks, hand_masks = [], []
for i, box in enumerate(face_boxes):
    mask = call_sam2(box, "face", i+1)
    if mask is not None:
        face_masks.append(mask)
    time.sleep(3)

for i, box in enumerate(hand_boxes):
    mask = call_sam2(box, "hand", i+1)
    if mask is not None:
        hand_masks.append(mask)
    time.sleep(3)

# ------------------- Visualize and Save -------------------
combined = np.zeros((height, width), dtype=np.uint8)
for m in face_masks + hand_masks:
    combined = cv2.bitwise_or(combined, m)

# Colored overlay
overlay = np.zeros_like(original_cv)
if face_masks:
    face_combined = np.zeros((height, width), dtype=np.uint8)
    for m in face_masks:
        face_combined = cv2.bitwise_or(face_combined, m)
    overlay[face_combined == 255] = [0, 255, 255]  # Yellow

if hand_masks:
    hand_combined = np.zeros((height, width), dtype=np.uint8)
    for m in hand_masks:
        hand_combined = cv2.bitwise_or(hand_combined, m)
    overlay[hand_combined == 255] = [255, 0, 255]  # Magenta

# Final blend
result = cv2.addWeighted(original_cv, 0.7, overlay, 0.3, 0)

# Contours
contour_img = original_cv.copy()
if face_masks:
    cnts, _ = cv2.findContours(face_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 255), 2)
if hand_masks:
    cnts, _ = cv2.findContours(hand_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, cnts, -1, (255, 0, 255), 2)

# Save
session_id = str(uuid.uuid4())[:6]
cv2.imwrite(f"FINAL_RESULT_{session_id}.png", result)
cv2.imwrite(f"FINAL_CONTOURS_{session_id}.png", contour_img)
cv2.imwrite(f"combined_mask_{session_id}.png", combined)

if face_masks:
    cv2.imwrite(f"faces_only_{session_id}.png", face_combined)
if hand_masks:
    cv2.imwrite(f"hands_only_{session_id}.png", hand_combined)

print("\n✅ DONE! Check your results:")
print(f"📁 FINAL_RESULT_{session_id}.png")
print(f"📁 FINAL_CONTOURS_{session_id}.png")
