import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ----------------------------------------------
# 0. Configuración de GPU y supresión de mensajes
# ----------------------------------------------
# Configurar uso limitado de la GPU (3 GB de memoria)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
        )
        print("GPU configurada con límite de 4 GB.")
    except RuntimeError as e:
        print(f"Error al configurar la GPU: {e}")
else:
    print("No se encontró ninguna GPU.")


# ----------------------------------------------
# 1. Configuración de rutas y carga de datos
# ----------------------------------------------
DATA_PATH = 'InfraredSolarModules'
METADATA_PATH = os.path.join(DATA_PATH, 'module_metadata.json')

# Verificar que el archivo existe
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"El archivo {METADATA_PATH} no existe. Verifica la ruta.")


df = pd.read_json(METADATA_PATH, orient='index').sort_index()
df['image_filepath'] = df.image_filepath.apply(lambda x: os.path.join(DATA_PATH, x))

# ----------------------------------------------
unique_values = df['anomaly_class'].value_counts()
print("\nUnique values and their frequency:")
print(unique_values)

# Set the figure size
plt.figure(figsize=(15, 6))

# Plot the bar chart
plt.bar(unique_values.index, unique_values.values)

# Set labels and title

plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('Frequency of Unique Values in Column')

# Show the plot
plt.show()

# ----------------------------------------------


# Verificar distribución inicial
print("Distribución de clases inicial:")
print(df['anomaly_class'].value_counts())

# ----------------------------------------------
# 2. División estratificada de datos
# ----------------------------------------------
crystal_df = df.query('anomaly_class != ("Hot-Spot", "Hot-Spot-Multi")')
crystal_df['image_filepath']
train_df, test_df = train_test_split(crystal_df, train_size=0.8, shuffle=True, random_state=1)

# Verificar la distribución después del split
print("Distribución en entrenamiento:")
print(train_df['anomaly_class'].value_counts())

print("Distribución en prueba:")
print(test_df['anomaly_class'].value_counts())

# Obtener todas las clases
all_classes = sorted(df.loc[~df['anomaly_class'].isin(["Hot-Spot", "Hot-Spot-Multi"]), 'anomaly_class'].unique())

# ----------------------------------------------
# 3. Generadores
# ----------------------------------------------
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2  # División interna para validación
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_dataset = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_filepath',
    y_col='anomaly_class',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='training',
    seed=42,
)

val_dataset = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_filepath',
    y_col='anomaly_class',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    subset='validation',
    shuffle=True,
    seed=42,
)

test_dataset = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_filepath',
    y_col='anomaly_class',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# ----------------------------------------------
# 5. Construcción del modelo
# ----------------------------------------------
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# ----------------------------------------------
# 6. Entrenamiento del modelo
# ----------------------------------------------
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1,
    callbacks=[callbacks],
)

# ----------------------------------------------
# 7. Evaluación del modelo
# ----------------------------------------------
results = model.evaluate(test_dataset, verbose=0)
print(f"Resultados en prueba: {results}")
print(f"Test Accuracy: {np.round(results[1] * 100,2)}%")

model.save('CNN_IR.h5')
# ----------------------------------------------
# 8. Reporte de clasificación
# ----------------------------------------------
predictions = np.argmax(model.predict(test_dataset), axis=1)
report= classification_report(test_dataset.labels, predictions, target_names=test_dataset.class_indices, zero_division=0)
print("Classification Report:\n", report)
