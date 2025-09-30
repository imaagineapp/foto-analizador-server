import cv2
import numpy as np
import os
from skimage.filters import sobel, laplace, median
from skimage.morphology import disk
from flask import Flask, request, jsonify

# Para detección de rostro y puntos faciales
import dlib
from imutils import face_utils

app = Flask(__name__)

# --- Cargamos detector y predictor ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # descargar de dlib

# ---------- FUNCIONES DE ANÁLISIS ----------

def sharpness_score(gray):
    lap = laplace(gray)
    sob = sobel(gray)
    return lap.var() + sob.var()

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
    saturation = hsv[:,:,1].mean() / 255.0
    return float(saturation)

def center_score(gray):
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    return float(center.var() / (gray.var() + 1e-6))

def file_size_score(path):
    size_kb = os.path.getsize(path) / 1024
    return float(min(1.0, size_kb / 100.0))

# --- Ojos abiertos ---
def eyes_open_score(gray):
    try:
        rects = detector(gray, 1)
        if not rects:
            return 0.0
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            # calcular apertura
            def eye_ratio(eye):
                vert = np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])
                hor = np.linalg.norm(eye[0]-eye[3])
                return vert / (hor + 1e-6)
            ratio = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2
            return float(min(1.0, ratio*5))  # normalizamos 0-1
    except:
        return 0.0

# --- Sonrisa ---
def smile_score(gray):
    try:
        rects = detector(gray, 1)
        if not rects:
            return 0.0
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[48:68]
            horiz = np.linalg.norm(mouth[0]-mouth[6])
            vert = np.linalg.norm(mouth[3]-mouth[9])
            ratio = vert / (horiz + 1e-6)
            return float(min(1.0, ratio*5))
    except:
        return 0.0

# ---------- FUNCIÓN PRINCIPAL ----------
def analyze_image(path, tipo="producto"):
    img = cv2.imread(path)
    if img is None:
        return {"error":"No se pudo cargar la imagen"},400

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

    # Métricas según tipo
    if tipo == "perfil":
        results["ojos_abiertos"] = eyes_open_score(gray)
    elif tipo == "redes":
        results["sonrisa"] = smile_score(gray)

    # Normalizar 0-1
    normalized = {k: min(1.0, max(0.0, v)) for k,v in results.items()}

    # Pesos por tipo
    weights = {
        "nitidez":0.2,
        "brillo":0.15,
        "contraste":0.1,
        "color":0.15,
        "encuadre":0.1,
        "peso":0.1
    }
    if tipo=="perfil":
        weights["ojos_abiertos"] = 0.2
    elif tipo=="redes":
        weights["sonrisa"] = 0.2

    # Puntaje ponderado
    score = sum(normalized[k]*weights.get(k,0) for k in normalized)

    return {"metricas":results, "normalizadas":normalized, "puntaje_final":float(score)},200

# ---------- RUTA FLASK ----------
@app.route("/analizar", methods=["POST"])
def analizar_endpoint():
    tipo = request.form.get("tipo","producto")  # perfil, redes, producto
    if "imagen" not in request.files:
        return jsonify({"error":"No se envió ningún archivo"}),400

    file = request.files["imagen"]
    if file.filename=="":
        return jsonify({"error":"Archivo sin nombre"}),400

    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    resultado,status = analyze_image(temp_path, tipo)
    os.remove(temp_path)
    return jsonify(resultado),status

# ---------- EJEMPLO LOCAL ----------
if __name__=="__main__":
    app.run(debug=True)
