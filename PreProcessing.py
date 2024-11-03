import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import pickle
import os

###########################################
############ Variables globales ###########
###########################################

cut_type = 'Vertical'  # Tipo de corte: 'Horizontal' o 'Vertical'
cut_factor = 0.5       # Factor de corte: 0.5 para la mitad vertical
min_distance = 90     # Distancia mínima entre líneas verticales
image_counter = 1      # Contador de imágenes
margin = 20            # Margen de recorte para las imágenes

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

        with open(mapx_path, 'rb') as mapx_file, open(mapy_path, 'rb') as mapy_file:
            mapx = pickle.load(mapx_file)
            mapy = pickle.load(mapy_file)

        return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def apply_double_crop(image: np.ndarray) -> tuple:
    height, width = image.shape[:2]
    if cut_type == 'Horizontal':
        cut_position = int(height * cut_factor)
        top_half = image[:cut_position, :]
        bottom_half = image[cut_position:, :]
        return top_half, bottom_half
    elif cut_type == 'Vertical':
        cut_position = int(width * cut_factor)
        left_half = image[:, :cut_position]
        right_half = image[:, cut_position:]
        return left_half, right_half
    else:
        print("Error: Tipo de corte no válido.")
        return image, image

def get_interpolated_points(x1, y1, x2, y2, num_points=10):
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int((1 - t) * x1 + t * x2)
        y = int((1 - t) * y1 + t * y2)
        points.append((x, y))
    return points

def filter_lines_by_distance(lines, min_distance):
    filtered_lines = []
    for line in lines:
        x1, _, x2, _ = line
        if not filtered_lines or abs((x1 + x2) // 2 - (filtered_lines[-1][0] + filtered_lines[-1][2]) // 2) >= min_distance:
            filtered_lines.append(line)
    return filtered_lines


###########################################
############## Carga de imágenes ##########
###########################################
rgb_dir = 'InputFiles/RGB'
ir_dir = 'InputFiles/IR'

# Obtener la lista de archivos en cada directorio
rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
ir_files = sorted([os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith('.tiff')])

if len(rgb_files) != len(ir_files):
    print("Error: Número de archivos en los directorios no coincide.")
    exit()

# Cargar las imágenes RGB e IR
rgb_images=[]
ir_images=[]
output_dir = "OutputImages"
os.makedirs(output_dir, exist_ok=True)

# Cargar la imagen IR
ir_image = tif.imread('InputFiles/IR/IR.tiff')
ir_image_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
ir_image_8bit = np.uint8(ir_image_normalized)
ir_image_undistorted = remove_distortion(ir_image_8bit, 'ir')

# Cargar la imagen RGB
rgb_image = cv2.imread('InputFiles/RGB/RGB.jpg')
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
rgb_image_undistorted = remove_distortion(gray_image, 'rgb')

# Recortar las imágenes en dos mitades y guardarlas
rgb_left_half, rgb_right_half = apply_double_crop(rgb_image_undistorted)
ir_left_half, ir_right_half = apply_double_crop(ir_image_undistorted)
cv2.imwrite(os.path.join(output_dir, 'RGB_left_half.png'), rgb_left_half)
cv2.imwrite(os.path.join(output_dir, 'RGB_right_half.png'), rgb_right_half)
cv2.imwrite(os.path.join(output_dir, 'IR_left_half.png'), ir_left_half)
cv2.imwrite(os.path.join(output_dir, 'IR_right_half.png'), ir_right_half)

rgb_images.append(rgb_left_half)
rgb_images.append(rgb_right_half)
ir_images.append(ir_left_half)
ir_images.append(ir_right_half)

###########################################
############# Procesamiento RGB ###########
###########################################


###########################################
############# Procesamiento IR ############
###########################################
output_dir = "OutputImages/Left"
os.makedirs(output_dir, exist_ok=True)
for image in ir_images:
    ir_blur = cv2.GaussianBlur(image, (3, 3), 0)
    ir_blur_output_path = os.path.join(output_dir, '000_IR_blur.png')
    cv2.imwrite(ir_blur_output_path, ir_blur)

    for clip_limit in range(1, 61, 5):
        for grid_size in [(4, 4), (8, 8)]:

            #CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            ir_image_clahe = clahe.apply(ir_blur)
            clahe_output_path = os.path.join(output_dir, f'{image_counter:03d}_CLAHE_clip_{clip_limit:02d}_grid_{grid_size[0]}x{grid_size[1]}.png')
            cv2.imwrite(clahe_output_path, ir_image_clahe)
            image_counter += 1

            #Otsu
            _, threshold_image = cv2.threshold(ir_image_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_image_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_threshold.png')
            cv2.imwrite(threshold_image_output_path, threshold_image)
            image_counter += 1

            #Canny
            edges = cv2.Canny(threshold_image, 100, 200)
            edges_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_edges.png')
            cv2.imwrite(edges_output_path, edges)
            image_counter += 1

            # Detección de líneas IR
            lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=90, minLineLength=450, maxLineGap=100)
            lines_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_lines.png')
            ir_lines = image.copy()
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(ir_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(lines_output_path, ir_lines)
            image_counter += 1

            # Filtrar líneas verticales en IR
            vertical_lines = []
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
                        if 85 <= angle <= 95:  # Umbral para líneas verticales
                            vertical_lines.append((x1, y1, x2, y2))

            vertical_lines_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_vertical_lines.png')
            ir_lines = image.copy()
            for x1, y1, x2, y2 in vertical_lines:
                cv2.line(ir_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(vertical_lines_output_path, ir_lines)
            image_counter += 1

            # Filtrar líneas verticales por distancia
            filtered_lines = filter_lines_by_distance(vertical_lines, min_distance)
            filtered_lines_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_filtered_lines.png')
            ir_lines = image.copy()
            for x1, y1, x2, y2 in filtered_lines:
                cv2.line(ir_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(filtered_lines_output_path, ir_lines)
            image_counter += 1

            if filtered_lines:
                x_coords = [x1 for x1, _, x2, _ in filtered_lines] + [x2 for _, _, x2, _ in filtered_lines]
                x_min, x_max = min(x_coords), max(x_coords)

                x_min = np.clip(x_min, 0, image.shape[1] - 1)
                x_max = np.clip(x_max, 0, image.shape[1] - 1)

                y_min, y_max = 0, image.shape[0]

                if x_min < x_max:
                    #Margen de recorte
                    x_min = max(0, x_min - margin)
                    x_max = min(image.shape[1], x_max + margin)

                    # Aplicar recorte
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    cropped_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_cropped.png')
                    cv2.imwrite(cropped_output_path, cropped_image)
                    image_counter += 1
                    print(f"    Imagen recortada y guardada en: {cropped_output_path}")
                else:
                    print(f"    Advertencia: x_min es igual o mayor que x_max. No se realizó el recorte. (image_counter={image_counter} output_dir={output_dir})")
                    print(f"    Valores de x en filtered_lines: {x_coords}")
            else:
                print(f"    Advertencia: No se encontraron líneas filtradas para recortar. (image_counter={image_counter})")
                x_coords = [x1 for x1, _, x2, _ in vertical_lines] + [x2 for _, _, x2, _ in vertical_lines]
                print(f"    Valores de x en vertical_lines: {x_coords}")


    output_dir = "OutputImages/Right"
    os.makedirs(output_dir, exist_ok=True)
    image_counter = 1