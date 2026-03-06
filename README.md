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

```bash
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
```

text
- 🏷️ Todas las imágenes están anotadas con bounding boxes de la clase `home`
- 📐 Resoluciones variables: `256x640` hasta `576x640`
- 🏷️ Las etiquetas están en **formato YOLO**
- 🏷️ Imagenes de entrenamiento **140 - 80%**
- 🏷️ Imagenes valid **34 - 20%**
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
```
```bash
python
from google.colab import drive
drive.mount('/content/drive')
```
---

## 4️⃣ Estructura del proyecto
```bash
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
```
---

## 5️⃣ Entrenamiento del modelo
***Comando de entrenamiento:***

```bash
!yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 project=runs/detect name=train

```
***Detalles del entrenamiento:***
🧠 Modelo base: yolov8n.pt (preentrenado en COCO)

⏱️ Épocas: 50

📏 Tamaño de imagen: 640x640

📂 Proyecto guardado en: runs/detect/train

***Resultados obtenidos:***

***Comportamiento de las métricas***

***Box Loss:*** empezó ~1.447 (Epoch 1) → bajó progresivamente a ~0.908 (Epoch 35)
Indica que el modelo aprende a predecir mejor las cajas.

***Cls Loss:*** empezó ~2.653 → bajó a ~0.8346
Las clases están aprendiendo a distinguirse, aunque tu dataset parece de 1 clase (nc=1), por lo que esta pérdida puede ser más sobre confianza que clasificación múltiple.

***DFL Loss:*** comenzó ~1.603 → bajó a ~1.186
Mejora la precisión de las coordenadas refinadas de las cajas.

***mAP50 y mAP50-95***

mAP50 empezó muy bajo (0.144 en Epoch 1) → mejoró hasta 0.614–0.579 en Epoch 33–34

mAP50-95 sigue más bajo (≈0.3), lo que indica que el modelo es razonablemente bueno detectando objetos a IoU 0.5, pero aún necesita mejorar precisión más estricta (IoU > 0.5).

***Precision (P) y Recall (R)***

P: osciló entre 0.3 y 0.7, indica qué tan confiables son las predicciones positivas.

R: llegó a ~0.66 en Epoch 32, significa que el modelo detecta ~66% de las casas reales.
✅ Mejor modelo guardado en: 

runs/detect/train10/weights/best.pt

---

## 6️⃣ Inferencia
***Inferencia en una imagen:***

```bash
!python src/inferencia.py --weights runs/detect/train10/weights/best.pt --source /content/imagen_prueba.jpg
Inferencia en una carpeta de imágenes:
```
```bash
!python src/inferencia.py --weights runs/detect/train/weights10/best.pt --source /content/dataset_roboflow/test/images

```

***Resultados obtenidos:***
📁 Salida guardada en: runs/detect/predict

⚡ Velocidad promedio: 23 ms por imagen

🏠 Casas detectadas por imagen: 2 a 5

🖼️ Imágenes generadas con bounding boxes

---

## 7️⃣ Evaluación del modelo
```bash
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

```
***Métricas de evaluación:***
```bash

Class     Images  Instances  P      R      mAP50  mAP50-95
all       29      77         0.72   0.584  0.608  0.349
```
***P (Precision)***: 0.72 → el porcentaje de predicciones correctas sobre todas las predicciones

***R (Recall):*** 0.584 → el porcentaje de objetos detectados sobre todos los objetos reales

***mAP50:*** 0.608 → mean Average Precision a IOU=0.5

***mAP50-95:*** 0.349 → promedio en varios thresholds de IOU

💡  el modelo está moderadamente preciso, pero hay margen de mejora (sobre todo si corriges las imágenes corruptas).

📊 Esta evaluación permite identificar falsos positivos y falsos negativos.
<img width="1357" height="918" alt="image" src="https://github.com/user-attachments/assets/b36b8779-46ad-4749-9e2b-3b8c7e669533" />

Deteccion Correcta, aun en un caso aparentemente dificil donde dos casas parecen una 


<img width="526" height="362" alt="image" src="https://github.com/user-attachments/assets/8e5a75da-1903-43c7-bab8-fef2a0754e08" />

Deteccion incorrecta, remonta cuadros y elige una puerta como una casa


---

## 8️⃣ Exportación del modelo
```bash
!yolo export model=runs/detect/train/weights/best.pt format=onnx

```
***Formatos soportados:***
📦 ONNX - Intercambio entre frameworks

📱 CoreML - Para dispositivos Apple

⚡ TensorRT - Optimizado para NVIDIA

🔧 OpenVINO - Para Intel

---

## 9️⃣ Recomendaciones y mejoras futuras
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

---

### 📫 Contacto
Autores:

👤 Jheansel Beltrán 

👤 Leonardo Fajardo 

📌 Proyecto académico - Taller de Visión Artificial y Detección de Objetos





