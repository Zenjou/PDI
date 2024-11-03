
# Aplicar Phase Correlation para alinear las imágenes
shift = cv2.phaseCorrelate(np.float32(ir_image_corrected), np.float32(rgb_resized_and_cropped))
shift_x, shift_y = shift[0]

# Aplicar la traslación calculada para alinear las imágenes
translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
aligned_rgb_image = cv2.warpAffine(rgb_image_corrected, translation_matrix, (rgb_resized_and_cropped.shape[1], rgb_resized_and_cropped.shape[0]))

# Mostrar las imágenes alineadas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ir_image_corrected, cmap='gray')
plt.title('Imagen IR Corregida')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(aligned_rgb_image, cmap='gray')
plt.title('Imagen RGB Alineada')
plt.axis('off')

plt.show()



##########################################################



align_image_1 = align_images(ir_image_corrected, rgb_resized, cv2.MOTION_TRANSLATION)
align_image_2 = align_images(ir_image_corrected, rgb_resized, cv2.MOTION_HOMOGRAPHY)

plt.figure(figsize=(12,12))
plt.subplot(3, 1, 1)
plt.imshow(ir_image_corrected, cmap='gray')
plt.title('Imagen IR Corregida')


plt.subplot(3, 1, 2)
plt.imshow(align_image_1, cmap='gray')
plt.title('Imagen RGB Alineada con Traslación')

plt.subplot(3, 1, 3)
plt.imshow(align_image_2, cmap='gray')
plt.title('Imagen RGB Alineada con Homografía')



plt.show()



# Recortar la imagen utilizando los bordes detectados
ir_recorted, bordes_ir = recortar_imagen_con_bordes(ir_image_corrected)
rgb_recorted, bordes_rgb = recortar_imagen_con_bordes(rgb_image_corrected)

# Debug
plt.figure(figsize=(10, 5))
plt.subplot(4, 1, 1)
plt.imshow(ir_recorted, cmap='gray')
plt.title('Imagen IR Recortada')


plt.subplot(4, 1, 2)
plt.imshow(bordes_ir, cmap='gray')
plt.title('Bordes IR')

plt.subplot(4, 1, 3)
plt.imshow(rgb_recorted, cmap='gray')
plt.title('Imagen RGB Recortada')

plt.subplot(4, 1, 4)
plt.imshow(bordes_rgb, cmap='gray')
plt.title('Bordes RGB')
plt.show()

ir_image_celsius = (ir_image * 0.04) - 273.15

plt.figure(figsize=(8, 6))
plt.imshow(ir_image_celsius, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('Temperatura (°C)', rotation=270, labelpad=15)
plt.axis('off')

plt.savefig('thermograma_con_barra.png', bbox_inches='tight', pad_inches=0)
plt.close()



        # Verificación de líneas y puntos de correspondencia
        if len(vertical_lines) >= 2:
            # Generar puntos interpolados en las líneas verticales
            ir_points_left = get_interpolated_points(*vertical_lines[0], num_points=10)
            ir_points_right = get_interpolated_points(*vertical_lines[1], num_points=10)
            ir_points = np.array(ir_points_left + ir_points_right, dtype=np.float32)

            # Generar puntos correspondientes en RGB usando las líneas laterales
            rgb_points_left = get_interpolated_points(left_line_rgb[0], left_line_rgb[1], left_line_rgb[2], left_line_rgb[3], num_points=10)
            rgb_points_right = get_interpolated_points(right_line_rgb[0], right_line_rgb[1], right_line_rgb[2], right_line_rgb[3], num_points=10)
            rgb_points = np.array(rgb_points_left + rgb_points_right, dtype=np.float32)

            # Calcular homografía y aplicar transformación
            H, _ = cv2.findHomography(ir_points, rgb_points)
            ir_warped = cv2.warpPerspective(ir_image_undistorted, H, (rgb_image_undistorted.shape[1], rgb_image_undistorted.shape[0]))

            ir_warped_output_path = os.path.join(output_dir, f'{image_counter:03d}_IR_Warped.png')
            cv2.imwrite(ir_warped_output_path, ir_warped)
            image_counter += 1
        else:
            print("Error: No se detectaron suficientes líneas verticales para calcular homografía.")


###########################################
############## Homografía #################
###########################################

gaussian = cv2.GaussianBlur(rgb_image_undistorted, (5, 5), 0)
gaussian_output_path = os.path.join(output_dir, '000_RGB_gaussian.png')
cv2.imwrite(gaussian_output_path, gaussian)

_, threshold_image = cv2.threshold(gaussian, 180, 255, cv2.THRESH_BINARY_INV)
threshold_output_path = os.path.join(output_dir, '000_RGB_threshold.png')
cv2.imwrite(threshold_output_path, threshold_image)

canny = cv2.Canny(threshold_image, 100, 200)
canny_output_path = os.path.join(output_dir, '000_RGB_canny.png')
cv2.imwrite(canny_output_path, canny)

# Detección de líneas en RGB
lines_rgb = cv2.HoughLinesP(canny, 0.5, np.pi / 360, threshold=90, minLineLength=200, maxLineGap=100)
vertical_lines_rgb = []
if lines_rgb is not None:
    for line in lines_rgb:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if 85 <= angle <= 95:
                vertical_lines_rgb.append((x1, y1, x2, y2))

#Dibujar líneas verticales en la imagen RGB (Sin filtrar)
rgb_lines_output_path = os.path.join(output_dir, '000_RGB_lines.png')
rgb_lines = rgb_image_undistorted.copy()
for x1, y1, x2, y2 in vertical_lines_rgb:
    cv2.line(rgb_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite(rgb_lines_output_path, rgb_lines)

# Filtrar líneas verticales en RGB
vertical_lines_rgb = filter_lines_by_distance(vertical_lines_rgb, min_distance)
left_line_rgb = vertical_lines_rgb[0]
right_line_rgb = vertical_lines_rgb[-1]

# Dibujar líneas verticales en la imagen RGB (Filtradas)
rgb_lines_filtered_output_path = os.path.join(output_dir, '000_RGB_lines_filtered.png')
rgb_lines_filtered = rgb_image_undistorted.copy()
for x1, y1, x2, y2 in vertical_lines_rgb:
    cv2.line(rgb_lines_filtered, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite(rgb_lines_filtered_output_path, rgb_lines_filtered)