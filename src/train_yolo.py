from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    data_yaml = root / "data.yaml"

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
    )

if __name__ == "__main__":
    main()