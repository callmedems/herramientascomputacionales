import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# paths de las imgs
img_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_q.jpg"

img2_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_2.jpg"

img3_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_3.jpg"

# Inicializar el lecto de EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Declaración de la función para procesar la img
def process_image(image_path, threshold_value, blur_value):
    '''
    Procesa la img de la placa aplicando umbralización, y luego un filtro gaussiano.
    Args: 
        image_path (str): ruta de la img a procesar.
        threshold_value (int): valor de umbral para la binarización.

    Return:
        img original, imagen umbralizada, imagen filtrada.
    '''
    img = cv.imread(image_path, cv.IMREAD.GRAYSCALE)
    _, thresh = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY_INV)
    gauss = cv.medianBlur(thresh, blur_value)
    return img, thresh, gauss

# Para mostrar las imágenes procesadas con matplotlib
def display(images, titles):
    '''
    Muestra las imágenes procesadas en una sola ventana.
    Args:
        images (list): lista de imgs a mostrar.
        titles (list): lista de títulos para cada imagen.
    '''
    plt.figure(figsize=(10, 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Para imprimir el output de la detección de texto en la placa
def detect_text(image, placa):
    '''
    Detecta el texto en la imagen procesada y lo imprime.
    Args:
        image: imagen procesada.
        placa (str): nombre de la placa.
    '''
    result = reader.readtext(image)
    print(f"Texto detectado en {placa}:")
    for detection in result:
        print(detection[1])

# Procesar y mostrar resultados para cada placa
for idx, (path, threshold) in enumerate([(img_path, 50), (img2_path, 70), (img3_path, 110)], start=1):
    img, thresh, gauss = process_image(path, threshold)
    display([img, thresh, gauss], ["Original", "Umbral", "Gauss"])
    detect_text(gauss, f"Placa {idx}")