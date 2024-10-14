import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import pickle
import os
import random
from deap import base, creator, tools, algorithms


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

# Cargar las imágenes IR
ir_image = tif.imread('InputFiles/IR.tiff')
ir_image_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
ir_image_8bit = np.uint8(ir_image_normalized)

# Cargar las imágenes RGB
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

############################################
######### Marcado de bordes (Líneas) #######
############################################

# Variables para almacenar las coordenadas iniciales y finales de las líneas
drawing = False  # True si el mouse está en movimiento
ix, iy = -1, -1
lines = []

# Función para dibujar una línea recta en la imagen
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, lines

    if event == cv2.EVENT_LBUTTONDOWN:  # Al hacer clic izquierdo
        drawing = True
        ix, iy = x, y  # Guardar el punto inicial de la línea

    elif event == cv2.EVENT_MOUSEMOVE:  # Mientras se arrastra el mouse
        if drawing:
            temp_image = ir_image_clahe.copy()
            cv2.line(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)  # Línea verde visible al arrastrar
            cv2.imshow('IR Image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # Al soltar el botón del mouse
        drawing = False
        # Guardar solo las coordenadas iniciales y finales para una línea recta
        lines.append((ix, iy, x, y))
        # Dibujar la línea recta final
        cv2.line(ir_image_clahe, (ix, iy), (x, y), (0, 255, 0), 2)  # Línea visible al soltar
        cv2.imshow('IR Image', ir_image_clahe)

# Mostrar la imagen y permitir al usuario dibujar manualmente las líneas rectas
cv2.namedWindow('IR Image')
cv2.setMouseCallback('IR Image', draw_line)

print("Dibuja manualmente las líneas de borde en la imagen. Presiona 'q' para salir.")

while True:
    cv2.imshow('IR Image', ir_image_clahe)

    # Salir cuando se presione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Convertir las líneas a una imagen binaria para la comparación
manual_edges_image = np.zeros_like(ir_image_clahe)
for line in lines:
    x1, y1, x2, y2 = line
    cv2.line(manual_edges_image, (x1, y1), (x2, y2), 255, 2)

# Guardar los bordes manuales
cv2.imwrite('manual_edges.png', manual_edges_image)

############################################
#### Optimización de los parámetros Canny ##
############################################

# Función de evaluación para el algoritmo genético
def evaluate_individual(individual):
    threshold1, threshold2 = sorted(individual)  # threshold1 < threshold2
    canny_edges = cv2.Canny(ir_image_clahe, threshold1, threshold2)
    difference = np.sum(np.abs(canny_edges.astype(np.float32) - manual_edges_image.astype(np.float32)))
    return difference,

# Crear la clase de optimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Definir el rango de los umbrales de Canny como genes de los individuos
toolbox.register("attr_threshold1", random.randint, 10, 200)  # Umbral inferior
toolbox.register("attr_threshold2", random.randint, 50, 300)  # Umbral superior

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_threshold1, toolbox.attr_threshold2), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def genetic_optimization(population_size=20, generations=30):
    population = toolbox.population(n=population_size)

    # Algoritmo genético con parámetros
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)

    # Encontrar el mejor individuo
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

############################################
######### Ejecutar el Algoritmo Genético ###
############################################
best_individual = genetic_optimization()

best_threshold1, best_threshold2 = sorted(best_individual)  # Asegurarse de que threshold1 < threshold2
print(f'Mejor configuración: Threshold1 = {best_threshold1}, Threshold2 = {best_threshold2}')

# Aplicar la mejor configuración de Canny
ir_best_canny_edges = cv2.Canny(ir_image_clahe, best_threshold1, best_threshold2)

# Mostrar la imagen de bordes resultante
plt.imshow(ir_best_canny_edges, cmap='gray')
plt.title(f'Bordes Optimizados (Threshold1: {best_threshold1}, Threshold2: {best_threshold2})')
plt.show()

# Guardar la imagen con los mejores parámetros
cv2.imwrite(f'best_canny_edges_ir.png', ir_best_canny_edges)
