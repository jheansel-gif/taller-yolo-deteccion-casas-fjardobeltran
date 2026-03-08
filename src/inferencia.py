import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO

from .utils import (
    anotar_resultado,
    extraer_detecciones,
    guardar_upload_temporal,
    validar_imagen,
)


APP_NAME = "API YOLO - Detección de Casas"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "runs/detect/train10/weights/best.pt",
)

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="API para cargar imágenes y detectar casas con YOLO",
)

model = None


@app.on_event("startup")
def load_model() -> None:
    global model

    ruta_modelo = Path(MODEL_PATH)
    if not ruta_modelo.exists():
        raise RuntimeError(f"No se encontró el modelo en: {ruta_modelo}")

    model = YOLO(str(ruta_modelo))


@app.get("/")
def root():
    return {
        "api": APP_NAME,
        "modelo": MODEL_PATH,
        "endpoints": {
            "GET /health": "Estado del servicio",
            "POST /predict/json": "Sube una imagen y devuelve detecciones en JSON",
            "POST /predict/image": "Sube una imagen y devuelve la imagen anotada",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict/json")
async def predict_json(
    archivo: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Umbral de confianza"),
    imgsz: int = Query(640, gt=0, le=2048, description="Tamaño de inferencia"),
):
    validar_imagen(archivo)
    ruta_temp = guardar_upload_temporal(archivo)

    try:
        results = model.predict(
            source=str(ruta_temp),
            save=False,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )

        detecciones = extraer_detecciones(results[0])

        return JSONResponse(
            content={
                "archivo": archivo.filename,
                "conf": conf,
                "imgsz": imgsz,
                "total_detecciones": len(detecciones),
                "detecciones": detecciones,
            }
        )
    finally:
        if ruta_temp.exists():
            ruta_temp.unlink()


@app.post("/predict/image")
async def predict_image(
    archivo: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Umbral de confianza"),
    imgsz: int = Query(640, gt=0, le=2048, description="Tamaño de inferencia"),
):
    validar_imagen(archivo)
    ruta_temp = guardar_upload_temporal(archivo)

    try:
        results = model.predict(
            source=str(ruta_temp),
            save=False,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )

        imagen_bytes = anotar_resultado(results[0])

        return StreamingResponse(
            BytesIO(imagen_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="pred_{archivo.filename}"'
            },
        )
    finally:
        if ruta_temp.exists():
            ruta_temp.unlink()