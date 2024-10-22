
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
