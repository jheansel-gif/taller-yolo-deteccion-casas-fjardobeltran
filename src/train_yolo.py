from ultralytics import YOLO

def main():

    # cargar modelo base
    model = YOLO("yolov8n.pt")

    # entrenar usando dataset
    model.train(
        data="/content/taller-yolo-deteccion-casas-fjardobeltran/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()