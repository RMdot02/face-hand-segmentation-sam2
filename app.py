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
import gradio as gr
from typing import Tuple, List, Optional
import tempfile

# ------------------- Configuration -------------------
class Config:
    MAX_DIM = 1024
    FACE_CONFIDENCE = 0.4
    HAND_CONFIDENCE = 0.3
    FACE_COLOR = [0, 255, 255]  # Yellow in BGR
    HAND_COLOR = [255, 0, 255]  # Magenta in BGR
    BLEND_ALPHA = 0.7
    OVERLAY_ALPHA = 0.3

# ------------------- Initialize Models and API -------------------
def initialize_models():
    """Initialize YOLO models and API token"""
    load_dotenv()
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    if not REPLICATE_API_TOKEN:
        raise ValueError("❌ REPLICATE_API_TOKEN not found in .env file!")
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    
    # Load models
    face_model = YOLO("yolov8n-face.pt")
    hand_model = YOLO("hand_yolov8n.pt")
    
    return face_model, hand_model

# Initialize models globally
try:
    face_model, hand_model = initialize_models()
    models_loaded = True
except Exception as e:
    print(f"Warning: Could not initialize models: {e}")
    models_loaded = False

# ------------------- Core Functions -------------------
def preprocess_image(image: Image.Image) -> Tuple[Image.Image, np.ndarray, str]:
    """Preprocess and resize image if needed"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large
    orig_width, orig_height = image.size
    if max(orig_height, orig_width) > Config.MAX_DIM:
        ratio = Config.MAX_DIM / max(orig_height, orig_width)
        new_w = int(orig_width * ratio)
        new_h = int(orig_height * ratio)
        image = image.resize((new_w, new_h))
    
    # Create a unique temporary file with proper extension
    temp_dir = tempfile.gettempdir()
    temp_filename = f"temp_image_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    # Save image
    image.save(temp_path, quality=95)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    return image, image_cv, temp_path

def detect_faces_and_hands(image_pil: Image.Image) -> Tuple[List, List]:
    """Detect faces and hands using YOLO models"""
    if not models_loaded:
        return [], []
    
    # Detect faces
    face_results = face_model(image_pil, verbose=False)
    face_boxes = []
    for i, box in enumerate(face_results[0].boxes.xyxy.cpu().numpy()):
        conf = face_results[0].boxes.conf[i].item()
        if conf > Config.FACE_CONFIDENCE:
            face_boxes.append(box)
    
    # Detect hands
    hand_results = hand_model(image_pil, verbose=False)
    hand_boxes = []
    for i, box in enumerate(hand_results[0].boxes.xyxy.cpu().numpy()):
        conf = hand_results[0].boxes.conf[i].item()
        if conf > Config.HAND_CONFIDENCE:
            hand_boxes.append(box)
    
    return face_boxes, hand_boxes

def call_sam2(box: np.ndarray, image_path: str, region_type: str, region_id: int, 
              height: int, width: int) -> Optional[np.ndarray]:
    """Call SAM2 API for segmentation"""
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        min_px = area * 0.1
        max_px = area * 3

        # Use context manager to ensure file is properly closed
        with open(image_path, "rb") as img_file:
            output = replicate.run(
                "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
                input={
                    "image": img_file,
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
            return None

        response = requests.get(str(mask_url), timeout=40)
        mask = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height))

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_px = np.sum(binary == 255)

        # Validate mask size
        if mask_px > max_px:
            roi_mask = np.zeros_like(binary)
            roi_mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
            roi_px = np.sum(roi_mask == 255)
            
            if min_px < roi_px < max_px or roi_px > 0:
                binary = roi_mask
            else:
                return None
        elif mask_px < min_px:
            return None

        # Clean mask
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary

    except Exception as e:
        print(f"SAM2 error: {e}")
        return None

def create_visualizations(original_cv: np.ndarray, face_masks: List, hand_masks: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create different visualization outputs"""
    height, width = original_cv.shape[:2]
    
    # Create colored overlay
    overlay = np.zeros_like(original_cv)
    
    # Face masks (Yellow)
    if face_masks:
        face_combined = np.zeros((height, width), dtype=np.uint8)
        for mask in face_masks:
            face_combined = cv2.bitwise_or(face_combined, mask)
        overlay[face_combined == 255] = Config.FACE_COLOR
    
    # Hand masks (Magenta)
    if hand_masks:
        hand_combined = np.zeros((height, width), dtype=np.uint8)
        for mask in hand_masks:
            hand_combined = cv2.bitwise_or(hand_combined, mask)
        overlay[hand_combined == 255] = Config.HAND_COLOR
    
    # Blended result
    result = cv2.addWeighted(original_cv, Config.BLEND_ALPHA, overlay, Config.OVERLAY_ALPHA, 0)
    
    # Contour visualization
    contour_img = original_cv.copy()
    if face_masks:
        cnts, _ = cv2.findContours(face_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, cnts, -1, Config.FACE_COLOR, 3)
    if hand_masks:
        cnts, _ = cv2.findContours(hand_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, cnts, -1, Config.HAND_COLOR, 3)
    
    # Combined mask
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in face_masks + hand_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return result, contour_img, combined_mask

def process_image(image: Image.Image, progress=gr.Progress()) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """Main processing function"""
    if not models_loaded:
        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, "Models not loaded!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        error_pil = Image.fromarray(cv2.cvtColor(error_img, cv2.COLOR_BGR2RGB))
        return error_pil, error_pil, error_pil, "❌ Models not initialized. Check your setup."
    
    try:
        progress(0.1, desc="Preprocessing image...")
        image_pil, original_cv, temp_path = preprocess_image(image)
        height, width = original_cv.shape[:2]
        
        progress(0.2, desc="Detecting faces and hands...")
        face_boxes, hand_boxes = detect_faces_and_hands(image_pil)
        
        detection_info = f"🔍 Detected: {len(face_boxes)} faces, {len(hand_boxes)} hands"
        
        if len(face_boxes) == 0 and len(hand_boxes) == 0:
            original_pil = Image.fromarray(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
            empty_mask = Image.fromarray(np.zeros_like(original_cv[:,:,0]))
            return original_pil, original_pil, empty_mask, detection_info + "\n⚠️ No faces or hands detected!"
        
        # Process faces
        face_masks = []
        total_regions = len(face_boxes) + len(hand_boxes)
        processed = 0
        
        for i, box in enumerate(face_boxes):
            progress(0.3 + (processed / total_regions) * 0.6, desc=f"Segmenting face {i+1}...")
            mask = call_sam2(box, temp_path, "face", i+1, height, width)
            if mask is not None:
                face_masks.append(mask)
            processed += 1
            time.sleep(1)  # Rate limiting
        
        # Process hands
        hand_masks = []
        for i, box in enumerate(hand_boxes):
            progress(0.3 + (processed / total_regions) * 0.6, desc=f"Segmenting hand {i+1}...")
            mask = call_sam2(box, temp_path, "hand", i+1, height, width)
            if mask is not None:
                hand_masks.append(mask)
            processed += 1
            time.sleep(1)  # Rate limiting
        
        progress(0.9, desc="Creating visualizations...")
        result, contour_img, combined_mask = create_visualizations(original_cv, face_masks, hand_masks)
        
        success_info = f"{detection_info}\n✅ Successfully segmented: {len(face_masks)} faces, {len(hand_masks)} hands"
        
        # Convert to PIL Images for Gradio
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        contour_pil = Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(combined_mask)
        
        progress(1.0, desc="Complete!")
        return result_pil, contour_pil, mask_pil, success_info
        
    except Exception as e:
        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)[:50]}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        error_pil = Image.fromarray(cv2.cvtColor(error_img, cv2.COLOR_BGR2RGB))
        return error_pil, error_pil, error_pil, f"❌ Error: {str(e)}"
    
    finally:
        # Always cleanup temp file
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass  # Ignore cleanup errors

# ------------------- Gradio Interface -------------------
def create_interface():
    """Create the Gradio interface"""
    
    # Clean, modern CSS with high contrast and visibility
    custom_css = """
    /* Global styling */
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0;
    }
    
    /* Feature boxes */
    .feature-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2563eb;
    }
    
    .feature-box h3, .feature-box h4 {
        color: #1f2937;
        margin-top: 0;
        font-weight: 600;
    }
    
    .feature-box ul {
        color: #4b5563;
        line-height: 1.6;
    }
    
    .feature-box li {
        margin-bottom: 0.5rem;
    }
    
    /* Info textbox styling - ensuring maximum visibility */
    .info-textbox,
    .info-textbox .wrap,
    .info-textbox .block {
        background-color: #ffffff !important;
        border: 2px solid #2563eb !important;
        border-radius: 8px !important;
    }
    
    .info-textbox textarea,
    .info-textbox .scroll-hide,
    .info-textbox input,
    .info-textbox .wrap textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        font-family: 'Inter', monospace !important;
        line-height: 1.5 !important;
        padding: 12px !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    /* Button styling */
    .gr-button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    .gr-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Tab styling */
    .tab-nav {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    
    .tab-nav button {
        background: transparent !important;
        color: #6b7280 !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .tab-nav button.selected {
        background: #2563eb !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Image containers */
    .gr-image {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .footer p {
        color: #6b7280;
        margin: 0.5rem 0;
    }
    
    .footer strong {
        color: #1f2937;
    }
    
    /* Override any Gradio dark mode styles */
    * {
        --body-text-color: #1f2937 !important;
        --block-background-fill: #ffffff !important;
        --block-border-color: #e5e7eb !important;
    }
    """
    
    # Create custom theme
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_radius="12px",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_border_color="*primary_600",
        button_primary_text_color="white",
        input_background_fill="white",
        input_border_color="*primary_600",
        input_border_width="2px",
    )
    
    with gr.Blocks(css=custom_css, title="🎭 Face & Hand Segmentation AI", theme=theme) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎭 AI Face & Hand Segmentation</h1>
            <p>Powered by YOLO Detection + SAM2 Segmentation</p>
            <p><em>Upload an image to automatically detect and segment faces and hands</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="feature-box"><h3>📤 Upload Image</h3></div>')
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400,
                    container=True
                )
                
                process_btn = gr.Button(
                    "🚀 Process Image", 
                    variant="primary", 
                    size="lg",
                    scale=1
                )
                
                gr.HTML("""
                <div class="feature-box">
                    <h4>✨ Features:</h4>
                    <ul>
                        <li>🎯 Automatic face detection</li>
                        <li>👋 Hand detection & segmentation</li>
                        <li>🎨 Multiple visualization modes</li>
                        <li>⚡ Powered by SAM2 AI</li>
                    </ul>
                </div>
                """)
                
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-box"><h3>📊 Results</h3></div>')
                
                info_output = gr.Textbox(
                    label="📋 Process Information",
                    lines=4,
                    interactive=False,
                    elem_classes=["info-textbox"],
                    value="📋 Upload an image and click 'Process Image' to get started!\n\n🎯 Status: Ready for processing\n💡 Tip: Works best with clear images of faces and hands",
                    show_copy_button=True,
                    container=True,
                    show_label=True
                )
                
                with gr.Tab("🎨 Segmented Result"):
                    result_output = gr.Image(
                        label="Segmented Image (Faces: Yellow, Hands: Magenta)",
                        height=400,
                        container=True
                    )
                
                with gr.Tab("📐 Contours"):
                    contour_output = gr.Image(
                        label="Contour Visualization",
                        height=400,
                        container=True
                    )
                
                with gr.Tab("⚫ Mask"):
                    mask_output = gr.Image(
                        label="Combined Binary Mask",
                        height=400,
                        container=True
                    )
        
        # Examples section
        gr.HTML('<div class="feature-box"><h3>🖼️ Try These Examples</h3><p>You can add example images from your project folder here for users to test with.</p></div>')
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[result_output, contour_output, mask_output, info_output],
            show_progress=True
        )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p><strong>🔧 Tech Stack:</strong> YOLO v8 • SAM2 • OpenCV • Gradio</p>
            <p><em>Built for automated face and hand segmentation with modern UI design</em></p>
        </div>
        """)
    
    return demo

# ------------------- Launch Application -------------------
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )