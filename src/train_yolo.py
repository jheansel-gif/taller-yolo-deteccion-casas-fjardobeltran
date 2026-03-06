from ultralytics import YOLO
import shutil
import os

def train():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,          # ajusta según GPU
        fliplr=0.5,       # flip horizontal
        degrees=10.0,     # rotaciones moderadas
        hsv_h=0.015,      # cambios de color
        hsv_s=0.7,
        hsv_v=0.4
    )

    # Guardar pesos finales en models/
    os.makedirs("models", exist_ok=True)
    shutil.copy(
        "runs/detect/train/weights/best.pt",
        "models/house-yolo.pt"
    )

if __name__ == "__main__":
    train()