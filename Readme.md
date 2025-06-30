# 🎭 Automated Face & Hand Segmentation using SAM2 API

This project implements a fully automated pipeline to detect **faces and hands** in an image and segment them using the **SAM2 (Segment Anything v2)** API by Meta. It combines the power of **YOLOv8** for detection and **SAM2** for fine-grained segmentation.

---

## 🧠 Features

- 🧍 Detect faces and hands from any image
- 🎯 Use SAM2 API (via Replicate) for precise segmentation
- 🎨 Overlay segmented masks with different colors
- 📐 Contour and binary mask visualization
- 🖼️ Gradio interface for easy image upload and preview
- 💡 Fully automated pipeline – no manual bounding boxes

---

## 🚀 Demo

![[output/Project Demo](<output/Project Demo>)]output/sample1/Segmented_result.png

---

## 🛠️ Installation

1. Clone the repository

git clone https://github.com/your-username/face-hand-segmentation-sam2.git
cd face-hand-segmentation-sam2



2. Create virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate



3. Install requirements
bash
Copy
Edit
pip install -r requirements.txt



4. Add your Replicate API key
Create a .env file and add:

env
Copy
Edit
REPLICATE_API_TOKEN=your_token_here

▶️ Run the App
bash
Copy
Edit
python app.py