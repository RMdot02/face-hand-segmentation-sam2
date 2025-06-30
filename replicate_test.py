import replicate
import os
from dotenv import load_dotenv

# Load API token from .env
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    raise ValueError("❌ API token missing. Please check your .env file.")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Input image URL
input = {
    "image": "https://replicate.delivery/pbxt/LMbGi83qiV3QXR9fqDIzTl0P23ZWU560z1nVDtgl0paCcyYs/cars.jpg",
    "use_m2m": True,
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
}

print("🔍 Testing SAM2 API with public image...")

output = replicate.run(
    "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
    input=input
)

print("✅ SAM2 API Output:")
print(output)
