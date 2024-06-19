import sys
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw

# check for correct usage
if len(sys.argv) != 2:
    print("Usage: python script.py <image.jpg>")
    sys.exit(1)

# get the image path from the command line arguments
image_path = sys.argv[1]

# construct the output image path
output_image_path = image_path.replace(".jpg", "_px.jpg")

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image = Image.open(image_path)
output = model(image)
results = Detections.from_ultralytics(output[0])

# draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for box in results.xyxy:
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)

# save the result image
image.save(output_image_path)
print(f"Result saved as {output_image_path}")

