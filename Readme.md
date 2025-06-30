🎭 Automated Face & Hand Segmentation using SAM2 API
This project implements a fully automated pipeline to detect faces and hands in images and segment them using the SAM2 (Segment Anything v2) API by Meta. It combines the power of YOLOv8 for detection and SAM2 for fine-grained segmentation.
🧠 Features

🧍 Automated Detection: Detect faces and hands from any image without manual annotations
🎯 SAM2 Integration: Use SAM2 API (via Replicate) for precise segmentation masks
🎨 Visual Overlays: Overlay segmented masks with different colors for clear visualization
📐 Multiple Outputs: Contour visualization, binary masks, and combined results
🖼️ Interactive UI: Gradio interface for easy image upload and real-time processing
💡 Fully Automated: No manual bounding boxes or user prompts required

🎥 Demo
output\Project Demo\demo.mp4

🛠️ Installation
Prerequisites

Python 3.8+
pip package manager
Replicate API account and token

Setup Steps

Clone the repository
bashgit clone https://github.com/RMdot02/face-hand-segmentation-sam2.git
cd face-hand-segmentation-sam2

Create virtual environment (recommended)
bashpython -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

Install dependencies
bashpip install -r requirements.txt

Set up API credentials
Create a .env file in the root directory:
envREPLICATE_API_TOKEN=your_replicate_api_token_here
Get your Replicate API token from: https://replicate.com/account/api-tokens

▶️ Usage
Method 1: Gradio Web Interface (Recommended)
bashpython app.py
Then open your browser and go to the local URL displayed (usually http://127.0.0.1:7860)
Method 2: Command Line Processing
bashpython main.py
This will process all images in the input_samples/ directory and save results to output/.
Method 3: Custom Processing
pythonfrom main import process_image

# Process a single image
result_path = process_image("path/to/your/image.jpg", "output/")
print(f"Results saved to: {result_path}")
🏗️ How It Works
Pipeline Overview

Face & Hand Detection

Uses YOLOv8 model trained specifically for face and hand detection
Detects bounding boxes with confidence scores
Filters detections based on confidence threshold (default: 0.5)


Prompt Generation

Converts bounding boxes to point prompts for SAM2
Uses center points of detected regions as positive prompts


SAM2 Segmentation

Sends image and prompts to Replicate SAM2 API
Receives high-quality segmentation masks
Processes masks for visualization


Visualization

Overlays masks with distinct colors (faces: red, hands: blue)
Generates contour visualizations
Saves multiple output formats



File Structure
face-hand-segmentation-sam2/
├── app.py                  # Gradio web interface
├── main.py                 # Main processing script
├── hand_yolov8n.pt        # YOLOv8 model for hand detection
├── yolov8n-face.pt        # YOLOv8 model for face detection
├── requirements.txt        # Python dependencies
├── .env                   # API credentials (create this)
├── input_samples/         # Input images
├── output/                # Generated results
│   ├── sample1/
│   ├── sample2/
│   └── Project Demo/
└── README.md
📋 Dependencies
Key libraries used:

ultralytics - YOLOv8 for object detection
replicate - SAM2 API integration
gradio - Web interface
opencv-python - Image processing
pillow - Image manipulation
numpy - Numerical operations

See requirements.txt for complete list.
🎯 Technical Details
Detection Models

Face Detection: YOLOv8n trained on face datasets
Hand Detection: YOLOv8n trained on hand datasets
Confidence Threshold: 0.5 (adjustable)

SAM2 Integration

API Provider: Replicate
Model: meta/sam-2:f3956b3b4b1d8c0bc63b62a2dafc1ad31815a4fb3f0e5a45b5aaf9b8d4d9e99e
Prompt Type: Point prompts (bounding box centers)

Output Formats

Detection visualization with bounding boxes
Segmentation masks overlaid on original image
Binary masks for each detected object
Contour visualizations

🚨 Limitations & Edge Cases
Current Limitations

API Dependency: Requires stable internet connection for Replicate API
Processing Time: SAM2 API calls can take 5-15 seconds per image
Detection Accuracy: Performance depends on image quality and lighting
Overlapping Objects: May struggle with heavily overlapping hands/faces

Known Edge Cases

Very small faces/hands (< 50px) may not be detected
Extreme poses or angles can reduce detection accuracy
Images with multiple people may have varying segmentation quality
API rate limits may affect batch processing

Potential Improvements

Add local SAM2 model support to reduce API dependency
Implement batch processing optimization
Add confidence score filtering for segmentation results
Support for video processing

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Meta AI for the SAM2 model
Ultralytics for YOLOv8 framework
Replicate for API hosting
Gradio for the web interface framework

📞 Contact
For questions or support, please open an issue on GitHub or contact the maintainer.