def calcular_calidad(imagen_path, tipo='producto'):
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

    # Ajustar ponderaciones según tipo de foto
    if tipo == 'perfil':
        # Nitidez y exposición en el rostro central
        h, w = gris.shape
        centro = gris[h//4:3*h//4, w//4:3*w//4]
        if np.mean(centro) < 50 or np.mean(centro) > 200:
            penalizacion += 1.5
        puntaje = (0.5*nitidez + 0.25*brillo + 0.25*contraste)/100

    elif tipo == 'producto':
        # En productos, priorizar brillo y exposición uniforme
        if gris.std() < 30:  # fondo muy uniforme o sombras
            penalizacion += 1
        puntaje = (0.3*nitidez + 0.4*brillo + 0.3*contraste)/100

    elif tipo == 'redes':
        # En redes, priorizar contraste y nitidez
        if contraste > 70:
            puntaje += 0.5  # bonus por buen contraste
        puntaje = (0.3*nitidez + 0.3*brillo + 0.4*contraste)/100

    else:
        # Default
        puntaje = (0.4*nitidez + 0.3*brillo + 0.3*contraste)/100

    # Puntaje final con penalizaciones
    puntaje = puntaje - penalizacion

    # Limitar de 0 a 10
    puntaje = min(max(puntaje, 0), 10)

    return round(puntaje, 2)
