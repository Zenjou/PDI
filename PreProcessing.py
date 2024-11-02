import cv2

# Cargar la imagen RGB 
imagen_rgb = cv2.imread('InputFiles/RGB.jpg')
gray_image = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2GRAY) 
box_blur = cv2.blur(gray_image,(4,4))
gaussian = cv2.GaussianBlur(gray_image,(7,7),0)

_,threshold_image = cv2.threshold(gaussian, 180, 255, cv2.THRESH_BINARY)
contorno,_ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos_dibujados = cv2.drawContours(imagen_rgb,contorno,-1,(0,255,0),3)

# Ordenar los contornos por área para seleccionar los paneles más grandes
contours = sorted(contorno, key=cv2.contourArea, reverse=True)

# Inicializar lista de recortes de los paneles
panel_crops = []

# Iterar sobre los dos contornos más grandes (asumimos que corresponden a los paneles grandes)
for i in range(2):
    # Obtener el rectángulo delimitador de cada contorno
    x, y, w, h = cv2.boundingRect(contours[i])
    
    # Recortar la imagen usando las coordenadas del rectángulo
    panel_crop = imagen_rgb[y:y+h, x:x+w]
    panel_crops.append(panel_crop)
    
    # Guardar cada recorte individualmente
    cv2.imwrite(f'panel_recorte_{i+1}.png', panel_crop)

# Mostrar el resultado final en pantalla para verificación
for idx, panel in enumerate(panel_crops):
    cv2.imshow(f'Panel {idx+1}', panel)


# Mostrar la imagen original con los contornos
# cv2.imshow('cont', contornos_dibujados)


cv2.waitKey(0)
cv2.destroyAllWindows()


