import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# paths de las imgs
img_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_q.jpg"

img2_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_2.jpg"

img3_path="D:\\demmi\\Documents\\CODING_VSCODE\\CODING_SCHOOL\\Python\\Works\\Act1_Filtrado_img\\herramientascomputacionales\\placa_3.jpg"

reader = easyocr.Reader(['en'], gpu=False)

# PLACA 1
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

_, thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)

gauss = cv.medianBlur(thresh, 15)

# resultados
imagenes = [img, thresh, gauss]
titulos = ["Original", "Umbral", "Gauss"]

plt.figure(figsize=(10, 4))
for i in range(len(imagenes)):
    plt.subplot(1, len(imagenes), i + 1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(titulos[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
    
result = reader.readtext(gauss)

#transcripcion
print("Texto detectado en Placa 1:")
for detection in result:
    print(detection[1])

# PLACA 2
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

_, thresh2 = cv.threshold(img2, 70, 255, cv.THRESH_BINARY_INV)

gauss2 = cv.medianBlur(thresh2, 15)


# resultados
imagenes2 = [img2, thresh2, gauss2]
titulos2 = ["Original", "Umbral", "Gauss "]

plt.figure(figsize=(10, 4))
for i in range(len(imagenes2)):
    plt.subplot(1, len(imagenes2), i + 1)
    plt.imshow(imagenes2[i], cmap='gray')
    plt.title(titulos2[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
    
result2 = reader.readtext(gauss2)

#imprimir transcripcion
print("Texto detectado en Placa 2:")
for detection in result2:
    print(detection[1])

# PLACA 3
img3 = cv.imread(img3_path, cv.IMREAD_GRAYSCALE)

_, thresh3 = cv.threshold(img3, 110, 255, cv.THRESH_BINARY_INV)

gauss3 = cv.medianBlur(thresh3, 15)


# resultados
imagenes3 = [img3, thresh3, gauss3]
titulos3 = ["Original", "Umbral", "Gauss "]

plt.figure(figsize=(10, 4))
for i in range(len(imagenes3)):
    plt.subplot(1, len(imagenes3), i + 1)
    plt.imshow(imagenes3[i], cmap='gray')
    plt.title(titulos3[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
    
result3 = reader.readtext(gauss3)

#imprimir transcripcion
print("Texto detectado en Placa 3:")
for detection in result3:
    print(detection[1])
