from ultralytics import YOLO

model = YOLO("models/house-yolo.pt")

results = model.predict(
    source="/content/drive/.shortcut-targets-by-id/1-zIxNE8KecruBVYgPscSJWWhMxIWiK2L/Taller2/dataset_roboflow/test/images/Screenshot-from-2026-03-01-15-23-16_png.rf.1256828f7a3f823791639dcf33973665.jpg",
    imgsz=640,
    conf=0.4,
    save=True
)

print("Inferencia completada")