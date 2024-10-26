import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import pickle
import os
import random

# Debug
show_intermediate_images = False

###########################################
############## Funciones ##################
###########################################

def remove_distortion(image: np.ndarray, image_type: str='rgb'):
    try:
        mapx_path = 'calibration files/RGB/mapx.pkl'
        mapy_path = 'calibration files/RGB/mapy.pkl'

        if image_type == 'ir':
            mapx_path = 'calibration files/IR/mapx.pkl'
            mapy_path = 'calibration files/IR/mapy.pkl'

        # Cargar archivos de mapas de distorsión
        with open(mapx_path, 'rb') as mapx_file, open(mapy_path, 'rb') as mapy_file:
            mapx = pickle.load(mapx_file)
            mapy = pickle.load(mapy_file)

        # Aplicar remapeo para corregir distorsión
        return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

############################################
############ Manejo de Imágenes ############
############################################

# Cargar las imágenes IR.
ir_image = tif.imread('InputFiles/IR.tiff')
ir_image_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
ir_image_8bit = np.uint8(ir_image_normalized)

# Cargar las imágenes RGB.
rgb_image = cv2.imread('InputFiles/RGB.jpg')
rgb_image_normalized = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
rgb_image_8bit = np.uint8(rgb_image_normalized)
gray_image = cv2.cvtColor(rgb_image_8bit, cv2.COLOR_BGR2GRAY)

# Corregir distorsión en las imágenes IR y RGB
ir_image_undistorted = remove_distortion(ir_image_8bit, 'ir')
rgb_image_undistorted = remove_distortion(gray_image, 'rgb')

# Directorio para guardar las imágenes
output_dir = "OutputImages"
os.makedirs(output_dir, exist_ok=True)

# Guardar las imágenes corregidas
ir_output_path = os.path.join(output_dir, '000_IR_Undistorted.png')
cv2.imwrite(ir_output_path, ir_image_undistorted)

if show_intermediate_images:
    # Mostrar las imágenes corregidas
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ir_image_undistorted, cmap='gray')
    plt.title('IR Image Undistorted')
    plt.axis('off')
    plt.show()

# Aplicar desenfoque Gaussiano
ir_Blur = cv2.GaussianBlur(ir_image_undistorted, (5,5), 0)
# Guardar la imagen desenfocada
ir_blur_output_path = os.path.join(output_dir, '000_IR_Blur.png')
cv2.imwrite(ir_blur_output_path, ir_Blur)

if show_intermediate_images:
    # Mostrar la imagen desenfocada
    plt.figure(figsize=(10, 5))
    plt.title('IR Image with Gaussian Blur')
    plt.imshow(ir_Blur, cmap='gray')
    plt.axis('off')
    plt.show()

# Pruebas con diferentes valores de clipLimit y tileGridSize
image_counter = 1  # Contador para asegurar nombres lineales de las imágenes
for clip_limit in range(1, 61, 5):  # Expandir rango de 1 a 60, incrementando en pasos de 5
    for grid_size in [(8, 8), (16, 16)]:
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        ir_image_clahe = clahe.apply(ir_Blur)

        # Guardar la imagen resultante de CLAHE
        clahe_output_path = os.path.join(output_dir, f'{image_counter:03d}_CLAHE_clip_{clip_limit:02d}_grid_{grid_size[0]}x{grid_size[1]}.png')
        cv2.imwrite(clahe_output_path, ir_image_clahe)
        image_counter += 1

        if show_intermediate_images:
            # Mostrar la imagen CLAHE
            plt.figure(figsize=(10, 5))
            plt.title(f'CLAHE with clipLimit={clip_limit}, tileGridSize={grid_size}')
            plt.imshow(ir_image_clahe, cmap='gray')
            plt.axis('off')
            plt.show()

        # Aplicar umbral (Threshold)
        _, threshold_image = cv2.threshold(ir_image_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_output_path = os.path.join(output_dir, f'{image_counter:03d}_Threshold_clip_{clip_limit:02d}_grid_{grid_size[0]}x{grid_size[1]}.png')
        cv2.imwrite(threshold_output_path, threshold_image)
        image_counter += 1

        if show_intermediate_images:
            # Mostrar la imagen umbralizada
            plt.figure(figsize=(10, 5))
            plt.title(f'Threshold with clipLimit={clip_limit}, tileGridSize={grid_size}')
            plt.imshow(threshold_image, cmap='gray')
            plt.axis('off')
            plt.show()

        #Operaciones morfológicas
        # Erosión para eliminar ruido
        eroded = cv2.erode(threshold_image, (5, 5), iterations=5)


        if show_intermediate_images:
            # Mostrar la imagen erosionada
            plt.figure(figsize=(10, 5))
            plt.title(f'Eroded with clipLimit={clip_limit}, tileGridSize={grid_size}')
            plt.imshow(eroded, cmap='gray')
            plt.axis('off')
            plt.show()

        # Aplicar Canny para detección de bordes
        edges = cv2.Canny(eroded, 100, 200)
        edges_output_path = os.path.join(output_dir, f'{image_counter:03d}_Edges_clip_{clip_limit:02d}_grid_{grid_size[0]}x{grid_size[1]}.png')
        cv2.imwrite(edges_output_path, edges)
        image_counter += 1

        if show_intermediate_images:
            # Mostrar la imagen de bordes
            plt.figure(figsize=(10, 5))
            plt.title(f'Edges with clipLimit={clip_limit}, tileGridSize={grid_size}')
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        # Detectar líneas usando la Transformada de Hough
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=30, minLineLength=50, maxLineGap=30)


        # Dibujar las líneas detectadas en la imagen original
        ir_image_contours = ir_image_undistorted.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(ir_image_contours, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja las líneas en verde

        # Guardar la imagen con líneas detectadas
        hough_output_path = os.path.join(output_dir, f'{image_counter:03d}_HoughLines_clip_{clip_limit:02d}_grid_{grid_size[0]}x{grid_size[1]}.png')
        cv2.imwrite(hough_output_path, ir_image_contours)
        image_counter += 1

        if show_intermediate_images:
            # Mostrar la imagen con líneas detectadas
            plt.figure(figsize=(10, 5))
            plt.title(f'Hough Lines with clipLimit={clip_limit}, tileGridSize={grid_size}')
            plt.imshow(ir_image_contours, cmap='gray')
            plt.axis('off')
            plt.show()
