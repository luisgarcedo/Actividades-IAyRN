# ==============================================================================
# PROYECTO: PUERTAS DINAMIXCAS CON CONTROL DE ACCESO
# IAyRN MATRICULAS: 2105438, 2100254, 2010528, 2178073, 2095105
# DESCRIPCIÓN: Red Neuronal Convolucional (CNN) entrenada desde cero con un
#              dataset propio para clasificar dos clases de lenguaje de señas.
# ==============================================================================
# Imporamos librerias. 
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Definimos el nombre de la carpeta raíz donde están guardadas nuestras subcarpetas.
DIR_DATOS = "datos"      

# Dimensiones estándar a las que se encogerán todas las imágenes para agilizar el procesamiento matemático
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Tamaño de lote (cuántas imágenes procesa la red antes de actualizar sus pesos internos)
BATCH_SIZE = 16          

# Épocas: El número de vueltas completas que dará la IA repasando el libro de fotos para aprender
EPOCHS = 10              

# Control de seguridad: Si el usuario olvidó crear la carpeta o se equivocó de ruta, el programa se detiene
if not os.path.exists(DIR_DATOS):
    print(f"Error: No se encuentra la carpeta '{DIR_DATOS}'. Revisa la estructura del proyecto.")
    exit()


# ==============================================================================
# FASE 1: CARGA, NORMALIZACIÓN Y DIVISIÓN DEL DATASET
# ==============================================================================
print("--- FASE 1: CARGANDO FOTOS DE LAS 2 LETRAS ---")

# ImageDataGenerator se encarga de preparar las fotos antes de meterlas a la red neuronal:
# 1. rescale=1./255 -> NORMALIZACIÓN: Convierte los píxeles (0-255) en decimales de 0.0 a 1.0 para evitar saturación.
# 2. validation_split=0.2 -> SEPARACIÓN: Aparta el 20% de las fotos para el examen final y deja el 80% para estudiar.
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generador de Entrenamiento: Lee las fotos del bloque de estudio (80%)
train_generator = datagen.flow_from_directory(
    DIR_DATOS,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Generador de Validación: Lee las fotos reservadas para el examen de calidad (20%)
val_generator = datagen.flow_from_directory(
    DIR_DATOS,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Extrae automáticamente las etiquetas detectadas basándose en los nombres de las carpetas físicas (['C', 'V'])
clases = list(train_generator.class_indices.keys())
num_clases = len(clases)
print(f"Letras detectadas en carpetas: {clases}")
print(f"Número de clases final a clasificar: {num_clases}")


# ==============================================================================
# FASE 2: ARQUITECTURA DE LA RED NEURONAL CONVOLUCIONAL (CNN)
# ==============================================================================
print("\n--- FASE 2: CONFIGURANDO LA RED NEURONAL ---")

# Creamos un modelo secuencial (los datos fluyen capa por capa en línea recta)
model = models.Sequential([
    # Capa Convolucional 1: Aplica 32 filtros analíticos para buscar bordes, curvas y líneas en la imagen de 64x64x3 (RGB)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    # Max Pooling 1: Reduce el tamaño a la mitad quedándose solo con los píxeles de características más importantes
    layers.MaxPooling2D((2, 2)),
    
    # Capa Convolucional 2: Aplica 64 filtros más profundos para combinar las líneas encontradas y entender formas complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Max Pooling 2: Vuelve a reducir el espacio para que el cálculo matemático sea ultra rápido
    layers.MaxPooling2D((2, 2)),
    
    # Capa de Aplanado (Flatten): Aplana la matriz bidimensional y la convierte en una sola fila larga de datos
    layers.Flatten(),
    
    # Capa Densa (Dense): Neuronas intermedias que razonan y asocian los patrones detectados con la respuesta correcta
    layers.Dense(64, activation='relu'),
    
    # Capa de Salida (Softmax): Devuelve los votos en forma de porcentajes probabilísticos distribuidos entre las clases (2 neuronas)
    layers.Dense(num_clases, activation='softmax') 
])

# Compilación del modelo: Le asignamos su optimizador matemático (adam) y la métrica para evaluar su rendimiento
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ==============================================================================
# FASE 3: PROCESO DE ENTRENAMIENTO EXPRÉS
# ==============================================================================
print("\n--- FASE 3: ENTRENANDO EL MODELO EN VIVO ---")

# Se ejecuta el ciclo de aprendizaje cruzando el generador de entrenamiento con el de validación
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
print("\n¡Modelo entrenado con éxito y alojado en la memoria RAM!")


# ==============================================================================
# FASE 4: CAPTURA, PROCESAMIENTO E INFERENCIA EN TIEMPO REAL (CÁMARA WEB)
# ==============================================================================
print("\n--- FASE 4: ABRIENDO CÁMARA (FILTRO ALTO) ---")
print("Colócate frente a la cámara. Presiona la tecla 'q' en la ventana de video para salir.")

# Inicializa la captura de video desde la cámara web nativa (índice 0)
cap = cv2.VideoCapture(0)

while True:
    # Captura el fotograma actual de la cámara
    ret, frame = cap.read()
    if not ret: 
        print("Error crítico: No se puede acceder a la señal de la cámara web.")
        break

    # Duplicamos el cuadro original para poder dibujarle texto e indicadores encima sin alterar la matriz de predicción
    display_frame = frame.copy()

    #---------------------------------------------------------------------------
    # ACONDICIONAMIENTO OBLIGATORIO (Debe ser idéntico al entrenamiento)
    #---------------------------------------------------------------------------
    # 1. Redimensionamos el fotograma actual exactamente a 64x64 píxeles
    img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # 2. OpenCV captura en formato BGR. Lo transformamos a RGB que es como trabaja TensorFlow
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Escalamos los píxeles dividiendo entre 255.0 para obtener valores flotantes de 0.0 a 1.0
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    
    # 4. Expandimos las dimensiones para simular un lote (pasa de tener forma [64,64,3] a [1,64,64,3])
    img_tensor = np.expand_dims(img_array, axis=0)

    #---------------------------------------------------------------------------
    # INFERENCIA MATEMÁTICA Y APLICACIÓN DE FILTRO DE CONFIANZA
    #---------------------------------------------------------------------------
    # Solicitamos al modelo que prediga la probabilidad del fotograma actual (verbose=0 apaga textos basura en terminal)
    prediccion = model.predict(img_tensor, verbose=0)
    
    # Buscamos cuál de las dos neuronas obtuvo el porcentaje más alto (Índice 0 o Índice 1)
    id_clase = np.argmax(prediccion[0])
    
    # Extraemos el valor real de esa probabilidad (ejemplo: 0.96)
    probabilidad = prediccion[0][id_clase]

    # FILTRO DE SEGURIDAD (92% de umbral): Impide que el modelo "adivine" con el fondo del cuarto o la ropa
    if probabilidad > 0.92:
        # Si supera el 92%, recuperamos el nombre real de la letra ('C' o 'V')
        letra_detectada = clases[id_clase]
        texto = f"Letra: {letra_detectada} ({probabilidad*100:.1f}%)"
        color_borde = (0, 255, 0) # El cuadro se ilumina en VERDE (Predicción válida)
    else:
        # Si la certeza es baja, el sistema entra en modo de espera protegiendo la pantalla de falsos positivos
        texto = "Buscando sena..."
        color_borde = (0, 0, 255) # El cuadro se queda en ROJO (Fondo o ruido visual)

    #---------------------------------------------------------------------------
    # DESPLIEGUE GRÁFICO EN INTERFAZ OPENCV
    #---------------------------------------------------------------------------
    # Dibujamos el texto dinámico y el cuadro indicador sobre nuestro fotograma duplicado
    cv2.putText(display_frame, texto, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_borde, 2)
    cv2.imshow("Proyecto IA desde Cero - FIME", display_frame)

    # Monitorea el teclado de la laptop. Si el usuario presiona la letra 'q', se rompe el bucle infinito
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberación de recursos de hardware y cierre ordenado del software
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")