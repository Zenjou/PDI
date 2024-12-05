# **ELO-328: Detección de Daños en Paneles Solares a través del Procesamiento Digital de Imágenes Aéreas RGB e IR**

## **Demos y Modelos (.ht) Disponibles**
Las demos y modelos entrenados (h.5) se encuentran disponibles en el Onedrive.

[Acceso al Onedrive](https://usmcl-my.sharepoint.com/:f:/g/personal/victor_munozs_usm_cl/ErIfns28g4hNiCxWcBhEl-wB8s4wONmhEIPHAOikkz4ORg?e=OyUMKX)

---

## **Manual de Instalación**

### **Requisitos previos**
1. **Entornos de Python**:
   - **Preprocesamiento y generación de informes**: Requiere un entorno Python estándar.
   - **Procesamiento con CNN**: Requiere un entorno Python con **CUDA activado**.

2. **Dependencias y rutas**:
   - Verificar las rutas de las dependencias y datasets antes de ejecutar los códigos.

---

### **Instalación**
1. **Descargar el repositorio desde GitHub**:
   Clonar el repositorio con:
   ```bash
   git clone https://github.com/Zenjou/PDI.git
   ```

2. **Descargar los modelos desde el Onedrive**:
   Los modelos necesarios están disponibles en el Onedrive.

3. **Instalación de dependencias**:
   Asegurarse de instalar las bibliotecas requeridas (detalladas en el archivo `requirements.txt` del repositorio).

---

## **Datasets necesarios**
- **Dataset de imágenes IR (entrenamiento CNN)**:
  [Infrared Solar Modules - Raptor Maps](https://github.com/RaptorMaps/InfraredSolarModules)

- **Dataset de imágenes RGB (entrenamiento CNN)**:
  [Solar Panel Images - Kaggle](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)

- **Dataset de imágenes RGB e IR (preprocesamiento)**:
  [Photovoltaic System Inspection - Kaggle](https://www.kaggle.com/datasets/marcosgabriel/photovoltaic-system-o-and-m-inspection)

---

## **Manual de Usuario**

### **Ejecución de módulos**
Debido a que los módulos no están integrados, deben ejecutarse de forma separada.

#### **Carpetas con los códigos correspondientes:**

- **Preprocesamiento IR**:
  Encargado de recortar las imágenes IR que recibe.
  - `PreProcesamiento-IR.py`

- **Preprocesamiento RGB**:
  Encargado de recortar las imágenes RGB que recibe.
  - `PreProcesamiento-RGB.py`

- **CNN RGB**:
  Carpeta con todo lo relacionado al entrenamiento y evaluación del modelo de clasificación con imágenes RGB:
  - `VGG16-Solar-Panel-fault-detection.ipynb`: Código de entrenamiento del modelo.
  - `predict_solar_panel.py`: Código para realizar pruebas con el modelo entrenado.

- **CNN IR**:
  Carpeta con todo lo relacionado al entrenamiento y evaluación del modelo de clasificación con imágenes IR:
  - `Resnet-50S-Solar-Panel-fault-detection.py`: Código de entrenamiento y pruebas del modelo entrenado.

- **Generador de informes**:
  Genera un informe en formato PDF basado en un archivo `.txt`:
  - `Informe.py`: Código para generar el informe en PDF.

---

### **Instrucciones generales**
1. Ejecutar los códigos mencionados según el módulo requerido.
2. Asegurarse de que las rutas de las dependencias y datasets estén correctamente configuradas antes de la ejecución.
3. Para la generación de informes, proporcionar un archivo `.txt` con la información relevante.

---

### **Warnings**
- Es **indispensable** verificar las rutas de las dependencias y datasets antes de ejecutar cualquier módulo.
- Se requiere un entorno Python con **CUDA activado** para ejecutar los códigos relacionados con las CNN.
