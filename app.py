import cv2
import numpy as np
import os
from skimage.filters import sobel, laplace, median
from skimage.morphology import disk
from flask import Flask, request, jsonify
import dlib
from imutils import face_utils

app = Flask(__name__)

# --- Cargamos detector y predictor ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ---------- FUNCIONES DE ANÁLISIS ----------
def sharpness_score(gray):
    return laplace(gray).var() + sobel(gray).var()

def brightness_score(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    total_pixels = gray.size
    dark_pixels = hist[:30].sum() / total_pixels
    bright_pixels = hist[220:].sum() / total_pixels
    return float(1 - (dark_pixels + bright_pixels))

def contrast_score(gray):
    p5, p95 = np.percentile(gray, (5,95))
    return float((p95 - p5) / 255.0)

def noise_score(gray):
    denoised = median(gray, disk(3))
    diff = np.abs(gray.astype("float32") - denoised.astype("float32"))
    return float(1 - (np.mean(diff) / 255.0))

def color_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:,:,1].mean() / 255.0)

def center_score(gray):
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    return float(center.var() / (gray.var() + 1e-6))

def file_size_score(path):
    size_kb = os.path.getsize(path) / 1024
    return float(min(1.0, size_kb / 100.0))

def eyes_open_score(gray):
    try:
        rects = detector(gray, 1)
        if not rects: return 0.0
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            def eye_ratio(eye):
                vert = np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])
                hor = np.linalg.norm(eye[0]-eye[3])
                return vert / (hor + 1e-6)
            ratio = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2
            return float(min(1.0, ratio*5))
    except: return 0.0

def smile_score(gray):
    try:
        rects = detector(gray, 1)
        if not rects: return 0.0
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[48:68]
            horiz = np.linalg.norm(mouth[0]-mouth[6])
            vert = np.linalg.norm(mouth[3]-mouth[9])
            ratio = vert / (horiz + 1e-6)
            return float(min(1.0, ratio*5))
    except: return 0.0

# ---------- FUNCIÓN PRINCIPAL ----------
def analyze_image(path, tipo="producto"):
    print("Ruta de imagen recibida:", path)
    print("Tipo:", tipo)

    img = cv2.imread(path)
    print("Imagen cargada:", img is not None)

    if img is None:
        # Siempre devolver métricas aunque sean 0
        empty_metrics = {
            "nitidez":0, "brillo":0, "contraste":0, "ruido":0,
            "color":0, "encuadre":0, "peso":0
        }
        if tipo == "perfil": empty_metrics["ojos_abiertos"] = 0
        elif tipo == "redes": empty_metrics["sonrisa"] = 0

        return {
            "metricas": empty_metrics,
            "normalizadas": empty_metrics,
            "puntaje_final": 0.0,
            "razon": "No se pudo analizar la foto",
            "mejor_foto": os.path.basename(path)
        }, 200

    # --- Procesar imagen ---
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

    if tipo == "perfil": results["ojos_abiertos"] = eyes_open_score(gray)
    elif tipo == "redes": results["sonrisa"] = smile_score(gray)

    normalized = {k: min(1.0, max(0.0, v)) for k, v in results.items()}

    weights = {
        "nitidez": 0.2, "brillo": 0.15, "contraste": 0.1,
        "color": 0.15, "encuadre": 0.1, "peso": 0.1
    }
    if tipo == "perfil": weights["ojos_abiertos"] = 0.2
    elif tipo == "redes": weights["sonrisa"] = 0.2

    score = sum(normalized[k] * weights.get(k, 0) for k in normalized)

    # Generar razón textual según métrica top
    top_metric = max(normalized, key=lambda k: normalized[k])
    razon_map = {
        "nitidez": "Mejor enfoque",
        "brillo": "Mejor iluminación",
        "contraste": "Mejor contraste",
        "ruido": "Menos ruido",
        "color": "Colores más vivos",
        "encuadre": "Mejor encuadre",
        "peso": "Tamaño de archivo óptimo",
        "ojos_abiertos": "Ojos abiertos",
        "sonrisa": "Sonrisa captada"
    }
    razon = razon_map.get(top_metric, f"Destaca en {top_metric}")

    # --- PRINTS PARA DEPURACIÓN ---
    print("Resultados calculados:", results)
    print("Normalizadas:", normalized)
    print("Puntaje final:", score)
    print("Razón:", razon)

    return {
        "metricas": results,
        "normalizadas": normalized,
        "puntaje_final": float(score),
        "razon": razon,
        "mejor_foto": os.path.basename(path)
    }, 200


# ---------- RUTA FLASK ----------
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

if __name__=="__main__":
    app.run(debug=True)
