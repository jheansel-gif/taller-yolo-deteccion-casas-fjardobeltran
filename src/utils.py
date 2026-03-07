from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List
import io

from fastapi import HTTPException, UploadFile
from PIL import Image

def mostrar_imagen(ruta):

    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.show()



FORMATOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

def validar_imagen(archivo: UploadFile) -> None:
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no soportado: {archivo.content_type}. "
                "Use JPEG, PNG, WEBP o BMP."
            ),
        )

    try:
        contenido = archivo.file.read()
        archivo.file.seek(0)

        imagen = Image.open(io.BytesIO(contenido))
        imagen.verify()
        archivo.file.seek(0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo subido no es una imagen válida: {str(e)}",
        )


def guardar_upload_temporal(archivo: UploadFile) -> Path:
    sufijo = Path(archivo.filename).suffix.lower() if archivo.filename else ".jpg"
    if not sufijo:
        sufijo = ".jpg"

    with NamedTemporaryFile(delete=False, suffix=sufijo) as temp:
        temp.write(archivo.file.read())
        archivo.file.seek(0)
        return Path(temp.name)


def extraer_detecciones(result: Any) -> List[Dict[str, Any]]:
    detecciones: List[Dict[str, Any]] = []

    if result.boxes is None:
        return detecciones

    nombres = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        detecciones.append(
            {
                "class_id": cls_id,
                "class_name": nombres.get(cls_id, str(cls_id)),
                "confidence": round(conf, 4),
                "bbox_xyxy": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            }
        )

    return detecciones


def anotar_resultado(result: Any) -> bytes:
    imagen_bgr = result.plot()
    imagen_rgb = Image.fromarray(imagen_bgr[:, :, ::-1])

    buffer = io.BytesIO()
    imagen_rgb.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()