import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from PIL import Image as PILImage
import os

# Función para obtener la fecha de creación de la imagen (usando os.stat)
def obtener_fecha_creacion(ruta_imagen):
    # Obtener los detalles del archivo (fecha de creación/modificación)
    stats = os.stat(ruta_imagen)
    # En Windows, st_ctime es la fecha de creación. En Unix, es la fecha de cambio de metadatos.
    return datetime.fromtimestamp(stats.st_ctime)

# Función para leer el contenido de data.txt
def leer_archivo_data(archivo, carpeta_imagenes):
    paneles = []
    with open(archivo, 'r') as f:
        lineas = f.readlines()
    
    # La primera línea es la imagen principal del campo solar
    imagen_principal = os.path.join(carpeta_imagenes, lineas[0].strip())
    
    # Las siguientes líneas contienen información de los paneles con fallas
    for linea in lineas[1:]:
        datos = linea.strip().split(';')  # Dividir la línea en nombre de imagen y tipo de falla
        nombre_imagen = datos[0]  # Ejemplo: panel_26.png
        tipo_falla = datos[1] if len(datos) > 1 else "Desconocida"
        numero_panel = nombre_imagen.split('_')[1].split('.')[0]  # Extraer el número del panel (ejemplo: "26")
        
        # Añadir la ruta completa de la imagen y los datos del panel
        paneles.append({
            "imagen": os.path.join(carpeta_imagenes, nombre_imagen),
            "numero_panel": numero_panel,
            "tipo_falla": tipo_falla
        })
    
    return imagen_principal, paneles

# Función para crear el PDF
def crear_reporte():
    doc = SimpleDocTemplate("reporte_panel_solar.pdf", pagesize=letter)
    elementos = []

    # Estilos
    estilos = getSampleStyleSheet()

    # Título
    titulo = Paragraph("Reporte de Panel Solar - Detección de Fallas", estilos['Title'])
    elementos.append(titulo)
    
    # Texto descriptivo
    descripcion = Paragraph("Este informe contiene los resultados del análisis de un campo solar a través de procesamiento digital de imágenes. A continuación se muestra el análisis de cada panel.", estilos['BodyText'])
    elementos.append(descripcion)
    elementos.append(Spacer(1, 12))  # Espacio en blanco

    # Leer la información desde "data.txt"
    archivo_data = "data.txt"  # Nombre del archivo que contiene los datos
    carpeta_imagenes = "imagenes"  # Carpeta donde se encuentran las imágenes
    imagen_principal, paneles = leer_archivo_data(archivo_data, carpeta_imagenes)

    # Obtener la fecha de creación de la imagen principal
    fecha_creacion_imagen = obtener_fecha_creacion(imagen_principal)
    fecha_formateada = fecha_creacion_imagen.strftime("%Y-%m-%d")
    hora_formateada = fecha_creacion_imagen.strftime("%H:%M:%S")

    # Extraer el ID del panel desde el nombre de la primera imagen (sin el .png)
    id_panel_principal = os.path.basename(imagen_principal).split('.')[0]

    # Contar las líneas para determinar si hay más de una, para detectar fallas
    fallas_detectadas = "Sí" if len(paneles) > 0 else "No"

    # Añadir la imagen principal del campo solar
    try:
        elementos.append(Image(imagen_principal, width=5*inch, height=3*inch))  # Ajusta el tamaño según sea necesario
    except Exception as e:
        print(f"Error al cargar la imagen principal: {e}")
        elementos.append(Paragraph("Error al cargar la imagen del campo solar.", estilos['BodyText']))
    
    # Añadir la tabla con los datos de la imagen principal
    datos_panel_principal = [
        ["ID del Panel", id_panel_principal],
        ["Fecha", fecha_formateada],  # Usar la fecha de creación de la imagen
        ["Hora", hora_formateada],  # Usar la hora de creación de la imagen
        ["Fallas Detectadas", fallas_detectadas],
    ]
    tabla_panel_principal = Table(datos_panel_principal)
    tabla_panel_principal.setStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige)])
    elementos.append(Spacer(1, 12))  # Espacio en blanco
    elementos.append(tabla_panel_principal)

    # Comprobar si hay paneles con fallas
    if len(paneles) > 0:
        # Agrupar los paneles en filas de 2 columnas
        panel_filas = []
        for i in range(0, len(paneles), 2):
            fila = []
            for j in range(2):
                if i + j < len(paneles):
                    panel = paneles[i + j]
                    # Crear la imagen del panel y la tabla con los datos del panel
                    imagen_panel = Image(panel["imagen"], width=1.25*inch, height=1.25*inch)  # Imagen más pequeña
                    datos_panel = [
                        ["ID del Panel", f"Panel {panel['numero_panel']}"],
                        ["Tipo de Falla", panel["tipo_falla"]],
                    ]
                    tabla_panel = Table(datos_panel)
                    tabla_panel.setStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                          ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                          ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                          ('FONTSIZE', (0, 0), (-1, -1), 9),
                                          ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                                          ('BACKGROUND', (0, 1), (-1, -1), colors.beige)])
                    # Añadir la imagen y la tabla como una celda de la fila
                    fila.append(Table([[imagen_panel, tabla_panel]], colWidths=[1.5*inch, 2*inch]))
                else:
                    # Añadir una celda vacía si no hay más paneles para llenar las dos columnas
                    fila.append('')

            # Añadir la fila completa a la lista de filas
            panel_filas.append(fila)

        # Crear una tabla con las filas de dos columnas por panel
        tabla_paneles_fallas = Table(panel_filas)
        tabla_paneles_fallas.setStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])  # Alineación superior de las celdas
        
        # Añadir la tabla al documento
        elementos.append(Spacer(1, 12))  # Espacio en blanco
        elementos.append(tabla_paneles_fallas)
    else:
        # Si no hay paneles con fallas, añadir un mensaje al documento
        elementos.append(Spacer(1, 12))  # Espacio en blanco

    # Generar el PDF
    doc.build(elementos)

# Ejecutar la creación del reporte
crear_reporte()
