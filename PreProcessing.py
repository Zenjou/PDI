import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import pickle

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

# CLAHE para mejorar el contraste de las imágenes IR y RGB
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
ir_image_clahe = clahe.apply(ir_image_undistorted)
rgb_image_clahe = clahe.apply(rgb_image_undistorted)


