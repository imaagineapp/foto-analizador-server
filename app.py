from flask import Flask, request, jsonify
from skimage import io, exposure, filters
import numpy as np
import cv2
import os

app = Flask(__name__)


def calcular_calidad(imagen_path):
    # Leer la imagen
    img = cv2.imread(imagen_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0

    # Convertir a gris
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nitidez (varianza del laplaciano)
    nitidez = cv2.Laplacian(gris, cv2.CV_64F).var()

    # Brillo promedio
    brillo = np.mean(gris)

    # Contraste
    contraste = gris.std()

    # ---- NUEVAS MÉTRICAS ----

    # Resolución mínima
    alto, ancho = img.shape[:2]
    penalizacion = 0
    if ancho < 720 or alto < 720:
        penalizacion += 2

    # Exposición (pixeles demasiado claros/oscuros)
    subexpuestos = np.mean(gris < 30) * 100
    sobreexpuestos = np.mean(gris > 220) * 100
    if subexpuestos > 40 or sobreexpuestos > 40:
        penalizacion += 2

    # Ruido digital (diferencia con suavizado)
    suavizada = cv2.GaussianBlur(gris, (3, 3), 0)
    ruido = np.mean(cv2.absdiff(gris, suavizada))
    if ruido > 25:
        penalizacion += 1.5

    # Puntaje final (ponderado + penalizaciones)
    puntaje = (0.4*nitidez + 0.3*brillo + 0.3*contraste) / 100
    puntaje = puntaje - penalizacion

    # Limitar de 0 a 10
    puntaje = min(max(puntaje, 0), 10)

    return round(puntaje, 2)


@app.route('/analizar', methods=['POST'])
def analizar():
    if 'foto' not in request.files:
        return jsonify({'error': 'No se envió foto'}), 400

    foto = request.files['foto']
    ruta = f"temp_{foto.filename}"
    foto.save(ruta)

    puntaje = calcular_calidad(ruta)

    # Borrar la foto temporal
    os.remove(ruta)

    return jsonify({'puntaje': puntaje})


if __name__ == '__main__':
    # Usar puerto dinámico que da Render, 5000 localmente
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
