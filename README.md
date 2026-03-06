# 🏠 Taller YOLO: Detección de Casas

## 👤 Autores
- **Jheansel Beltrán**
- **Leonardo Fajardo**

> Proyecto realizado en el marco del Taller de Visión Artificial y Detección de Objetos.

---

## 1️⃣ Descripción del proyecto
Este proyecto implementa un modelo de detección de casas utilizando **YOLOv8** (You Only Look Once) de Ultralytics. El objetivo es identificar casas en imágenes urbanas, marcar su posición mediante **bounding boxes** y generar resultados visuales junto con métricas de desempeño.

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

text
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

python
from google.colab import drive
drive.mount('/content/drive')
4️⃣ Estructura del proyecto
text
taller-yolo-deteccion-casas/
├── data.yaml                          # Configuración del dataset YOLO
├── models/                             # Modelos personalizados (si aplica)
├── src/
│   ├── train.py                        # Script de entrenamiento
│   └── inferencia.py                    # Script de inferencia
├── requirements.txt                     # Dependencias del proyecto
├── README.md
└── runs/
    └── detect/                          # Resultados de entrenamiento e inferencia
        ├── train/                        # Carpeta generada durante entrenamiento
        │   └── weights/
        │       ├── best.pt                # Mejor modelo entrenado
        │       └── last.pt                 # Último modelo guardado
        └── predict/                       # Resultados de inferencia
5️⃣ Entrenamiento del modelo
Comando de entrenamiento:

bash
!yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 project=runs/detect name=train
Detalles del entrenamiento:

🧠 Modelo base: yolov8n.pt (preentrenado en COCO)

⏱️ Épocas: 50

📏 Tamaño de imagen: 640x640

📂 Proyecto guardado en: runs/detect/train

Resultados obtenidos:

Métrica	Valor
mAP50	0.85
Precision	0.88
Recall	0.81
✅ Mejor modelo guardado en: runs/detect/train/weights/best.pt

6️⃣ Inferencia
Inferencia en una imagen:

bash
!python src/inferencia.py --weights runs/detect/train/weights/best.pt --source /content/imagen_prueba.jpg
Inferencia en una carpeta de imágenes:

bash
!python src/inferencia.py --weights runs/detect/train/weights/best.pt --source /content/dataset_roboflow/test/images
Resultados obtenidos:

📁 Salida guardada en: runs/detect/predict

⚡ Velocidad promedio: 23 ms por imagen

🏠 Casas detectadas por imagen: 2 a 5

🖼️ Imágenes generadas con bounding boxes

7️⃣ Evaluación del modelo
bash
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
Métricas de evaluación:

mAP50 - Precisión media con IoU=0.50

mAP50-95 - Precisión media en diferentes IoUs

Precision - Proporción de detecciones correctas

Recall - Capacidad de detectar todas las casas presentes

📊 Esta evaluación permite identificar falsos positivos y falsos negativos.

8️⃣ Exportación del modelo
bash
!yolo export model=runs/detect/train/weights/best.pt format=onnx
Formatos soportados:

📦 ONNX - Intercambio entre frameworks

📱 CoreML - Para dispositivos Apple

⚡ TensorRT - Optimizado para NVIDIA

🔧 OpenVINO - Para Intel

9️⃣ Recomendaciones y mejoras futuras
📸 Incrementar el dataset para mejorar detección de casas pequeñas

🔄 Data augmentation: rotación, escalado, flipping horizontal/vertical

🎯 Ajuste de hiperparámetros: learning rate, batch size, optimizer

🎥 Extender a video: detección en tiempo real o secuencias urbanas

🗺️ Aplicación en mapas: detectar viviendas en imágenes satelitales

🔗 Referencias
Ultralytics YOLOv8 Documentation

Roboflow Universe - Datasets públicos

Redmon et al., You Only Look Once: Unified, Real-Time Object Detection, CVPR 2016

Google Colab - Guía oficial

📫 Contacto
Autores:

👤 Jheansel Beltrán - jheansel.beltran@example.com

👤 Leonardo Fajardo - leonardo.fajardo@example.com

📌 Proyecto académico - Taller de Visión Artificial y Detección de Objetos



