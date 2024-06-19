import sys
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw, ImageFilter

# Prüfe die Eingabeparameter
if len(sys.argv) != 2:
    print("Usage: python script.py <image.jpg>")
    sys.exit(1)

# Lade den Bildpfad aus den Kommandozeilenargumenten
image_path = sys.argv[1]

# Erstelle den Ausgabepfad für das Bild
output_image_path = image_path.replace(".jpg", "_px.jpg")

# Lade das Modell herunter
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Lade das Modell
model = YOLO(model_path)

# Funktion zum Verpixeln eines Bildbereichs
def pixelate_area(image, box, pixel_size=9):
    x1, y1, x2, y2 = map(int, box)
    face_area = image.crop((x1, y1, x2, y2))
    face_area = face_area.resize(
        (face_area.width // pixel_size, face_area.height // pixel_size), Image.NEAREST
    )
    face_area = face_area.resize((x2 - x1, y2 - y1), Image.NEAREST)
    image.paste(face_area, (x1, y1))

# Führe die Bildverarbeitung durch
image = Image.open(image_path)
output = model(image)
results = Detections.from_ultralytics(output[0])

# Verpixele die erkannten Gesichter
for box in results.xyxy:
    pixelate_area(image, box)

# Speichere das Ergebnisbild
image.save(output_image_path)
print(f"Result saved as {output_image_path}")

