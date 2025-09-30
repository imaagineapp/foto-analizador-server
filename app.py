import cv2
import numpy as np
import os
from skimage.filters import sobel, laplace, median
from skimage.morphology import disk

# ---------- FUNCIONES DE ANÁLISIS ----------

# 1. Nitidez (Laplaciano + Sobel)
def sharpness_score(gray):
    lap = laplace(gray)
    sob = sobel(gray)
    return (lap.var() + sob.var())

# 2. Brillo (histograma)
def brightness_score(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    total_pixels = gray.size
    dark_pixels = hist[:30].sum() / total_pixels
    bright_pixels = hist[220:].sum() / total_pixels
    return float(1 - (dark_pixels + bright_pixels))  # penaliza extremos

# 3. Contraste (percentiles)
def contrast_score(gray):
    p5, p95 = np.percentile(gray, (5,95))
    return float((p95 - p5) / 255.0)

# 4. Ruido (filtro mediana)
def noise_score(gray):
    denoised = median(gray, disk(3))
    diff = np.abs(gray.astype("float32") - denoised.astype("float32"))
    return float(1 - (np.mean(diff) / 255.0))

# 5. Colores (saturación promedio)
def color_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean() / 255.0
    return float(saturation)

# 6. Encuadre (detalle en el centro vs toda la foto)
def center_score(gray):
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    return float(center.var() / (gray.var() + 1e-6))

# 7. Tamaño de archivo
def file_size_score(path):
    size_kb = os.path.getsize(path) / 1024
    return float(min(1.0, size_kb / 100.0))  # hasta 100 KB puntaje completo


# ---------- FUNCIÓN PRINCIPAL ----------

def analyze_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = {
        "nitidez": sharpness_score(gray),
        "brillo": brightness_score(gray),
        "contraste": contrast_score(gray),
        "ruido": noise_score(gray),
        "color": color_score(img),
        "encuadre": center_score(gray),
        "peso": file_size_score(path)
    }

    # Normalizar cada métrica a 0–1
    normalized = {k: min(1.0, max(0.0, v)) for k,v in results.items()}

    # Promedio de puntaje final
    score = np.mean(list(normalized.values()))

    return {
        "metricas": results,
        "normalizadas": normalized,
        "puntaje_final": float(score)
    }


# ---------- EJEMPLO DE USO ----------
if __name__ == "__main__":
    ruta = "ejemplo.jpg"  # reemplazar con una imagen real
    analisis = analyze_image(ruta)
    print(analisis)
