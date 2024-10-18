import cv2

# Cargar la imagen RGB 
imagen_rgb = cv2.imread('InputFiles/RGB.jpg')
gray_image = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2GRAY) 
box_blur = cv2.blur(gray_image,(4,4))
gaussian = cv2.GaussianBlur(gray_image,(7,7),0)

_,threshold_image = cv2.threshold(gaussian, 180, 255, cv2.THRESH_BINARY)
contorno,_ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos_dibujados = cv2.drawContours(imagen_rgb,contorno,-1,(0,255,0),3)

# Mostrar la imagen original con los contornos
cv2.imshow('gauss',gaussian)
cv2.imshow('cont', contornos_dibujados)
cv2.imshow('thr',threshold_image)


# esperar a que se pulse una tecla y cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()


