import cv2
import numpy as np
import os
from skimage.filters import sobel, median
from skimage.morphology import disk
from scipy.ndimage import laplace
from flask import Flask, request, jsonify

# pylint: disable=E1101

app = Flask(__name__)

# ----------------------------------------------------
#                 FUNCIONES DE ANÁLISIS
# ----------------------------------------------------

def sharpness_score(gray):
    gray_float = gray.astype(np.float32) / 255.0
    return laplace(gray_float).var() + sobel(gray_float).var()

def brightness_score(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size
    dark_pixels = hist[:30].sum() / total_pixels
    bright_pixels = hist[220:].sum() / total_pixels
    return 1.0 - float(dark_pixels + bright_pixels)

def contrast_score(gray):
    p5, p95 = np.percentile(gray, (5, 95))
    return float((p95 - p5) / 255.0)

def noise_score(gray):
    denoised = median(gray, disk(3))
    diff = np.abs(gray.astype("float32") - denoised.astype("float32"))
    return 1.0 - float(np.mean(diff) / 255.0)

def color_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean() / 255.0)

def center_score(gray):
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    return float(center.var() / (gray.var() + 1e-6))

def file_size_score(path):
    size_kb = os.path.getsize(path) / 1024
    return float(min(1.0, max(0.0, size_kb / 100.0)))


# ----------------------------------------------------
#                FUNCIÓN PRINCIPAL
# ----------------------------------------------------

def analyze_image(path, tipo="producto"):
    img = cv2.imread(path)
    if img is None:
        empty_metrics = {k: 0.0 for k in [
            "nitidez", "brillo", "contraste", "ruido", "color", "encuadre", "peso"]}
        return {
            "tipo": tipo,
            "metricas": empty_metrics,
            "normalizadas": empty_metrics,
            "puntaje_final": 0.0,
            "razon": "No se pudo analizar la foto",
            "mejor_foto": os.path.basename(path)
        }, 200

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = {
        "nitidez": float(sharpness_score(gray)),
        "brillo": float(brightness_score(gray)),
        "contraste": float(contrast_score(gray)),
        "ruido": float(noise_score(gray)),
        "color": float(color_score(img)),
        "encuadre": float(center_score(gray)),
        "peso": float(file_size_score(path))
    }

    normalized = {k: float(min(1.0, max(0.0, v))) for k, v in results.items()}

    # ---------- PESOS SEGÚN TIPO ----------
    if tipo == "producto":
        weights = {
            "nitidez": 0.25, "brillo": 0.15, "contraste": 0.1,
            "ruido": 0.1, "color": 0.15, "encuadre": 0.2, "peso": 0.05
        }
    elif tipo == "perfil":
        weights = {
            "nitidez": 0.15, "brillo": 0.25, "contraste": 0.1,
            "ruido": 0.1, "color": 0.25, "encuadre": 0.1, "peso": 0.05
        }
    elif tipo == "red_social":
        weights = {
            "nitidez": 0.15, "brillo": 0.2, "contraste": 0.2,
            "ruido": 0.1, "color": 0.25, "encuadre": 0.05, "peso": 0.05
        }
    else:
        weights = {
            "nitidez": 0.25, "brillo": 0.15, "contraste": 0.1,
            "ruido": 0.1, "color": 0.15, "encuadre": 0.15, "peso": 0.1
        }

    weighted_contributions = {
        k: float(normalized[k] * weights.get(k, 0)) for k in normalized}
    score = float(sum(weighted_contributions.values()))

    top_metric = max(weighted_contributions, key=weighted_contributions.get)

    razon_map = {
        "nitidez": "Tiene un mejor enfoque y detalles más definidos.",
        "brillo": "Posee una iluminación más equilibrada.",
        "contraste": "Muestra un contraste más nítido entre luces y sombras.",
        "ruido": "Presenta menos ruido digital y una imagen más limpia.",
        "color": "Destaca por sus colores más vivos y naturales.",
        "encuadre": "Está mejor encuadrada y centrada.",
        "peso": "Tiene un tamaño de archivo óptimo (buena calidad sin exceso de peso)."
    }
    razon = razon_map.get(
        top_metric, f"Destaca en {top_metric.replace('_', ' ')}")

    return {
        "tipo": tipo,
        "metricas": results,
        "normalizadas": normalized,
        "puntaje_final": score,
        "razon": razon,
        "mejor_foto": os.path.basename(path)
    }, 200


# ----------------------------------------------------
#                  ENDPOINT 1 FOTO
# ----------------------------------------------------

@app.route("/analizar", methods=["POST"])
def analizar_endpoint():
    tipo = request.form.get("tipo", "producto")

    if "foto" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["foto"]
    if file.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400

    temp_path = f"temp_{file.filename}"
    try:
        file.save(temp_path)
        resultado, status = analyze_image(temp_path, tipo)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(resultado), status


# ----------------------------------------------------
#          ENDPOINT MULTIPLE (3 FOTOS JUNTAS)
# ----------------------------------------------------

@app.route("/analizar-multiples", methods=["POST"])
def analizar_multiples_endpoint():
    tipo = request.form.get("tipo", "producto")

    if "fotos" not in request.files:
        return jsonify({"error": "No se enviaron fotos[]"}), 400

    files = request.files.getlist("fotos")
    if len(files) == 0:
        return jsonify({"error": "Lista vacía en fotos[]"}), 400

    resultados = []
    temp_paths = []

    try:
        # Guardar temporales
        for idx, file in enumerate(files):
            temp_path = f"temp_multi_{idx}_{file.filename}"
            file.save(temp_path)
            temp_paths.append(temp_path)

        # Analizar cada imagen
        for path in temp_paths:
            data, _ = analyze_image(path, tipo)
            resultados.append(data)

    finally:
        # Borrar temporales
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    return jsonify({"resultados": resultados}), 200


# ----------------------------------------------------
#                     PING
# ----------------------------------------------------

@app.route("/ping")
def ping():
    return "pong", 200


# ----------------------------------------------------
#                     MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
