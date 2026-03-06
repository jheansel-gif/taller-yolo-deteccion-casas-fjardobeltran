from ultralytics import YOLO

def main():

    # Ruta al modelo entrenado
    model = YOLO("/content/taller-yolo-deteccion-casas-fjardobeltran/runs/detect/train3/weights/best.pt")

    # Carpeta con imágenes de prueba
    source = "/content/drive/MyDrive/Colab Notebooks/Applied/Taller2/dataset_roboflow/test/images"

    # Ejecutar inferencia
    results = model.predict(
        source=source,
        save=True,
        conf=0.25
    )

    print("Inferencia completada")

if __name__ == "__main__":
    main()