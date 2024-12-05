# Pre-procesamiento imágenes RGB 
# Autor: Javiera Peña
import cv2

# Cargar la imagen RGB -> imagen cargada como matriz de 3 canales 
imagen_rgb = cv2.imread('InputFiles/RGB.jpg')

gray_image = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2GRAY) # se convierte la imagen BGRA a escala de grises para poder trabajar con filtros 

# aplicación de filtros que reducen el ruido para mejorar la detección de contornos
box_blur = cv2.blur(gray_image,(4,4))
gaussian = cv2.GaussianBlur(gray_image,(7,7),0)

new_imagergb = imagen_rgb.copy() # nueva imagen para el recorte -> esto se hace para no modificar la imagen original en caso de necesitarla
#bcv2.imshow('imagen',new_imagergb) 

_,threshold_image = cv2.threshold(gaussian, 180, 255, cv2.THRESH_BINARY) # aplicar un umbral binario a la imagen Guassiana
contorno,_ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos_dibujados = cv2.drawContours(imagen_rgb,contorno,-1,(0,255,0),3)  # crear los contornos en imagen rgb

# ordenar los contornos por área para seleccionar los paneles más grandes
contours = sorted(contorno, key=cv2.contourArea, reverse=True)

# inicializar lista de recortes de los paneles
panel_crops = []

# iterar sobre los dos contornos más grandes (asumimos que corresponden a los paneles grandes)
for i in range(2):
    # obtener el rectángulo delimitador de cada contorno
    x, y, w, h = cv2.boundingRect(contours[i])
    
    # recortar la imagen usando las coordenadas del rectángulo
    panel_crop = new_imagergb[y:y+h, x:x+w] # se usa la nueva imagen para recortar 
    panel_crops.append(panel_crop)
    
    # guardar cada recorte individualmente
    cv2.imwrite(f'panel_recorte_{i+1}.png', panel_crop)

# mostrar el resultado final en pantalla para verificación
for idx, panel in enumerate(panel_crops):
    cv2.imshow(f'Panel {idx+1}', panel)


# mostrar la imagen original con los contornos
# cv2.imshow('cont', contornos_dibujados)


cv2.waitKey(0)
cv2.destroyAllWindows()


