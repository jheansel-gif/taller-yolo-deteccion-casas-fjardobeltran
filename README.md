# 🏠 Taller YOLO: Detección de Casas

## 👤 Autores

- **Jheansel Beltrán**
- **Leonardo Fajardo**

> Proyecto realizado en el marco del Taller de Visión Artificial y Detección de Objetos.

---

## 1️⃣ Descripción del proyecto

Este proyecto implementa un modelo de detección de casas utilizando **YOLOv8** (You Only Look Once) de Ultralytics.  
El objetivo es identificar casas en imágenes urbanas, marcar su posición mediante **bounding boxes** y generar resultados visuales junto con métricas de desempeño.

**El flujo completo del proyecto incluye:**

- 📦 Preparación del dataset
- 🧠 Entrenamiento del modelo YOLO
- 🔍 Inferencia sobre imágenes nuevas
- 📊 Evaluación del modelo (mAP, Precision, Recall)
- 📁 Exportación y análisis de resultados

---

## 2️⃣ Dataset

El dataset proviene de **Roboflow**, organizado de la siguiente manera:


dataset_roboflow/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/


- 🏷️ Todas las imágenes están anotadas con bounding boxes de la clase `home`
- 📐 Resoluciones variables: `256x640` hasta `576x640`
- 🏷️ Las etiquetas están en **formato YOLO**
- 🖼️ Cantidad de imágenes de prueba procesadas en inferencia: **10**
- 🏠 Resultado promedio: **2–5 casas detectadas por imagen**

---

## 3️⃣ Instalación y entorno

Se recomienda usar **Google Colab** con GPU activa (Tesla T4) y **Python 3.12**.

**Requisitos:**

```bash
pip install ultralytics==8.4.19
pip install torch==2.10.0+cu128
pip install pandas numpy matplotlib

Montaje de Google Drive (para dataset):
from google.colab import drive
drive.mount('/content/drive')

## 4️⃣ Estructura del proyecto

taller-yolo-deteccion-casas/
├── data.yaml                 # Configuración del dataset YOLO
├── models/                   # Modelos personalizados (si aplica)
├── src/
│   ├── train.py              # Script de entrenamiento
│   └── inferencia.py         # Script de inferencia
├── requirements.txt          # Dependencias del proyecto
├── README.md
└── runs/
    └── detect/               # Resultados de entrenamiento e inferencia

## 5️⃣ Entrenamiento del modelo
Comando de entrenamiento:

!yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 project=runs/detect name=train

⚙️ yolov8n.pt es el modelo base preentrenado

⏳ Se entrenó durante 50 epochs

📏 Las imágenes se redimensionaron a 640x640

Resultados del entrenamiento:

🏆 Mejor modelo guardado en: runs/detect/train/weights/best.pt

📈 Métricas de desempeño (ejemplo):

mAP50: 0.85

Precision: 0.88

Recall: 0.81

## 6️⃣ Inferencia
Inferencia en una imagen:

!python src/inferencia.py --weights runs/detect/train/weights/best.pt --source /content/imagen_prueba.jpg

Inferencia en una carpeta de imágenes:
!python src/inferencia.py --weights runs/detect/train/weights/best.pt --source /content/drive/MyDrive/Colab\ Notebooks/Applied/Taller2/dataset_roboflow/test/images

esultados:

📁 Guardados en: runs/detect/predict

⚡ Velocidad promedio: 23 ms por imagen

🏠 Detecciones por imagen: 2–5 casas

📦 Bounding boxes generadas para cada casa

## 7️⃣ Evaluación del modelo
Se recomienda evaluar con:

!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

Métricas obtenidas: mAP50, mAP50-95, Precision, Recall
📊 Permite comparar desempeño y detectar falsos positivos/negativos.

## 8️⃣ Exportación del modelo
Opcional para producción o deployment:
!yolo export model=runs/detect/train/weights/best.pt format=onnx
Genera modelos en formatos ONNX, CoreML, TensorRT, entre otros.

## 9️⃣ Recomendaciones y mejoras futuras
📸 Incrementar dataset para mejorar detección de casas pequeñas

🔄 Aplicar data augmentation (rotación, escalado, flipping) para aumentar robustez

🎥 Implementar detección en video o en mapas urbanos

⚙️ Ajustar hiperparámetros (learning rate, batch size) según el tamaño del dataset

🔗 Referencias
Ultralytics YOLOv8 Documentation

Roboflow

Redmon et al., You Only Look Once: Unified, Real-Time Object Detection, CVPR 2016

📫 Contacto
Autor: Jheansel Hasler Beltrán


