# TAREA 7 - Machine Learning
# OPERACIÓN 01: Red Neuronal con Fashion MNIST
# OPERACIÓN 02: Regresión Lineal Simple

# =============================================================================
# OPERACIÓN 01: Crear y entrenar una red neuronal con Python y TensorFlow
# =============================================================================

print("=" * 70)
print("OPERACIÓN 01: RED NEURONAL CON FASHION MNIST")
print("=" * 70)

# Paso 1: Importar las bibliotecas necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

print(f"\nTensorFlow versión: {tf.__version__}")

# Paso 2: Cargar y Explorar el Conjunto de Datos Fashion MNIST
print("\n--- Paso 2: Cargando el dataset Fashion MNIST ---")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Etiquetas de entrenamiento: {y_train.shape}")
print(f"Datos de prueba: {X_test.shape}")
print(f"Etiquetas de prueba: {y_test.shape}")

# Definir las etiquetas de las categorías
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Visualizar algunas imágenes del dataset
print("\n--- Visualizando muestras del dataset ---")
plt.figure(figsize=(14, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_train[i], cmap="binary")
    plt.title(labels[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()

# Paso 3: Normalización de Datos
print("\n--- Paso 3: Normalizando los datos ---")
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0
print("Datos normalizados al rango [0, 1]")

# Paso 4: Definición de la Arquitectura de la Red Neuronal
print("\n--- Paso 4: Definiendo la arquitectura de la red neuronal ---")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Mostrar resumen del modelo
model.summary()

# Paso 5: Compilar el Modelo
print("\n--- Paso 5: Compilando el modelo ---")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Modelo compilado exitosamente")

# Paso 6: Entrenar el Modelo
print("\n--- Paso 6: Entrenando el modelo (30 épocas) ---")
history = model.fit(
    X_train_norm, 
    y_train, 
    epochs=30, 
    validation_data=(X_test_norm, y_test),
    verbose=1
)

# Visualizar el progreso del entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.grid(True)

plt.tight_layout()
plt.show()

# Paso 7: Evaluar el Modelo
print("\n--- Paso 7: Evaluando el modelo ---")
test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Pérdida en el conjunto de prueba: {test_loss:.4f}')

# Hacer algunas predicciones de ejemplo
print("\n--- Predicciones de ejemplo ---")
predictions = model.predict(X_test_norm[:10])

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap="binary")
    predicted_label = labels[np.argmax(predictions[i])]
    true_label = labels[y_test[i]]
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"Real: {true_label}\nPredicho: {predicted_label}", color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()

# =============================================================================
# OPERACIÓN 02: Implementar y graficar regresión lineal simple
# =============================================================================

print("\n" + "=" * 70)
print("OPERACIÓN 02: REGRESIÓN LINEAL SIMPLE")
print("=" * 70)

# Importar bibliotecas adicionales
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist

print("\n--- Análisis de relación entre Temperatura y Ventas ---")

# Datos de ejemplo (ficticios)
temperatura = np.array([15, 16, 18, 20, 21, 23, 25, 27, 30, 32])
ventas = np.array([500, 520, 560, 580, 600, 640, 680, 700, 760, 800])

print(f"Datos de temperatura: {temperatura}")
print(f"Datos de ventas: {ventas}")

# Dividir los datos en conjunto de entrenamiento y prueba
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    temperatura.reshape(-1, 1), 
    ventas,
    test_size=0.2, 
    random_state=42
)

print(f"\nDatos de entrenamiento: {len(X_train_reg)}")
print(f"Datos de prueba: {len(X_test_reg)}")

# Crear y entrenar el modelo de regresión lineal
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

print(f"\n--- Parámetros del modelo ---")
print(f"Coeficiente (pendiente): {reg_model.coef_[0]:.2f}")
print(f"Intercepto: {reg_model.intercept_:.2f}")
print(f"Ecuación: Ventas = {reg_model.coef_[0]:.2f} * Temperatura + {reg_model.intercept_:.2f}")

# Hacer predicciones
y_pred_train = reg_model.predict(X_train_reg)
y_pred_test = reg_model.predict(X_test_reg)

# Evaluación del modelo
r_sq = reg_model.score(X_test_reg, y_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_test)
rmse = np.sqrt(mse)

print(f"\n--- Métricas de evaluación ---")
print(f'Coeficiente de determinación (R²): {r_sq:.4f}')
print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Raíz del error cuadrático medio (RMSE): {rmse:.2f}')

# Gráfica de resultados
plt.figure(figsize=(10, 6))
plt.scatter(temperatura, ventas, color="blue", s=100, alpha=0.6, label="Datos reales", zorder=3)
plt.plot(temperatura, reg_model.predict(temperatura.reshape(-1, 1)), 
         color="red", linewidth=2, label="Línea de regresión", zorder=2)
plt.scatter(X_test_reg, y_test_reg, color="green", s=150, marker='s', 
            alpha=0.8, label="Datos de prueba", zorder=4)
plt.xlabel("Temperatura (°C)", fontsize=12)
plt.ylabel("Ventas", fontsize=12)
plt.title("Regresión Lineal: Temperatura vs Ventas", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Análisis de residuos
residuos = y_test_reg - y_pred_test
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_test, residuos, color='purple', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuos, bins=5, color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# APLICACIONES PRÁCTICAS ADICIONALES
# =============================================================================

print("\n" + "=" * 70)
print("APLICACIONES PRÁCTICAS EN LA INDUSTRIA DE LA MODA")
print("=" * 70)

# -----------------------------------------------------------------------------
# OPCIÓN: Clasificar TUS PROPIAS IMÁGENES
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🖼️ CLASIFICAR TUS PROPIAS IMÁGENES")
print("=" * 70)
print("\nPuedes subir tus propias imágenes de ropa para clasificar:")
print("1. Ejecuta la celda de código de abajo")
print("2. Haz clic en 'Choose Files' y selecciona una imagen")
print("3. El modelo clasificará automáticamente la prenda\n")

# Código para subir y clasificar imagen propia
try:
    from google.colab import files
    from PIL import Image
    import io
    
    print("📤 Sube una imagen de ropa (formato: JPG, PNG):")
    uploaded = files.upload()
    
    if uploaded:
        # Procesar cada imagen subida
        for filename, file_data in uploaded.items():
            print(f"\n--- Procesando: {filename} ---")
            
            # Cargar y preprocesar la imagen
            img = Image.open(io.BytesIO(file_data))
            
            # Mostrar imagen original
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Imagen Original", fontweight='bold')
            plt.axis('off')
            
            # Convertir a escala de grises y redimensionar a 28x28
            img_gray = img.convert('L')
            img_resized = img_gray.resize((28, 28))
            img_array = np.array(img_resized)
            
            plt.subplot(1, 3, 2)
            plt.imshow(img_resized, cmap='binary')
            plt.title("Procesada (28x28)", fontweight='bold')
            plt.axis('off')
            
            # Normalizar y preparar para predicción
            img_normalized = img_array / 255.0
            img_input = img_normalized.reshape(1, 28, 28)
            
            # Hacer predicción
            prediction = model.predict(img_input, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Mostrar predicción
            plt.subplot(1, 3, 3)
            plt.barh(labels, prediction[0], color='skyblue')
            plt.xlabel('Probabilidad', fontweight='bold')
            plt.title('Predicción del Modelo', fontweight='bold')
            plt.xlim(0, 1)
            
            # Destacar la predicción más alta
            max_idx = np.argmax(prediction[0])
            plt.barh(labels[max_idx], prediction[0][max_idx], color='green')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n🎯 RESULTADO:")
            print(f"   Categoría predicha: {labels[predicted_class]}")
            print(f"   Confianza: {confidence:.2f}%")
            
            if confidence < 70:
                print(f"   ⚠️ Baja confianza - La imagen podría no ser clara o no pertenecer a estas categorías")
            
            print("\n📊 Top 3 predicciones:")
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            for i, idx in enumerate(top_3_idx, 1):
                print(f"   {i}. {labels[idx]:15s} - {prediction[0][idx]*100:.2f}%")
    else:
        print("❌ No se subió ninguna imagen")
        
except ImportError:
    print("ℹ️ Esta funcionalidad solo está disponible en Google Colab")
    print("   Si estás en Colab, continúa con las aplicaciones de demostración abajo.")
    print("   Si estás en local, puedes modificar esta sección para cargar imágenes locales.")

print("\n" + "=" * 70)

# -----------------------------------------------------------------------------
# APLICACIÓN 1: Búsqueda Visual por Similitud de Imagen
# -----------------------------------------------------------------------------
print("\n--- APLICACIÓN 1: Búsqueda Visual por Similitud Real ---")
print("Encontrar productos visualmente similares usando características de la red\n")

# Seleccionar una imagen de prueba como "búsqueda"
search_image_idx = 42
search_image = X_test_norm[search_image_idx]
search_label_real = labels[y_test[search_image_idx]]

print(f"🔍 Producto de búsqueda: {search_label_real}")

# MÉTODO 1: Extracción de características con la red neuronal
# Creamos un modelo que extrae las características de la capa oculta
# Primero necesitamos construir el modelo si no ha sido llamado
if not model.built:
    model.build(input_shape=(None, 28, 28))

# Crear extractor de características usando la arquitectura interna
feature_extractor = keras.Sequential([
    model.layers[0],  # Flatten
    model.layers[1]   # Dense de 128 neuronas
])

print("Extrayendo características visuales de todas las imágenes...")
# Extraer características de la imagen de búsqueda
search_features = feature_extractor.predict(search_image.reshape(1, 28, 28), verbose=0)

# Extraer características de todas las imágenes de prueba (en lotes para eficiencia)
print("Procesando conjunto de prueba...")
all_features = feature_extractor.predict(X_test_norm, verbose=0, batch_size=256)

# Calcular similitud usando distancia euclidiana
from scipy.spatial.distance import cdist

# Calcular distancias entre la imagen de búsqueda y todas las demás
distances = cdist(search_features, all_features, metric='euclidean')[0]

# Ordenar por similitud (menor distancia = más similar)
similar_indices = np.argsort(distances)[1:7]  # Excluir la imagen misma (índice 0)
similarities = 1 / (1 + distances[similar_indices])  # Convertir distancia a similitud

print(f"✓ Búsqueda completada usando características de la red neuronal\n")

# MÉTODO 2 (Adicional): Similitud de píxeles directa
print("Método alternativo: Similitud directa de píxeles...")
pixel_distances = np.sum(np.abs(X_test_norm - search_image), axis=(1, 2))
pixel_similar_indices = np.argsort(pixel_distances)[1:7]

plt.figure(figsize=(16, 8))

# === MÉTODO 1: Búsqueda por Características (Red Neuronal) ===
plt.subplot(2, 7, 1)
plt.imshow(X_test[search_image_idx], cmap="binary")
plt.title(f"BÚSQUEDA\n{search_label_real}", fontweight='bold', color='blue', fontsize=10)
plt.axis("off")
plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                   edgecolor='blue', linewidth=3))

for i, idx in enumerate(similar_indices):
    plt.subplot(2, 7, i+2)
    plt.imshow(X_test[idx], cmap="binary")
    
    item_label = labels[y_test[idx]]
    similarity_score = similarities[i] * 100
    
    # Color según si es la misma categoría o no
    is_same_category = y_test[idx] == y_test[search_image_idx]
    border_color = 'green' if is_same_category else 'orange'
    title_color = 'green' if is_same_category else 'darkorange'
    
    plt.title(f"{item_label}\nSim: {similarity_score:.1f}%", 
              fontsize=8, color=title_color)
    plt.axis("off")
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                       edgecolor=border_color, linewidth=2))

# === MÉTODO 2: Búsqueda por Píxeles ===
plt.subplot(2, 7, 8)
plt.imshow(X_test[search_image_idx], cmap="binary")
plt.title(f"BÚSQUEDA\n{search_label_real}", fontweight='bold', color='purple', fontsize=10)
plt.axis("off")
plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                   edgecolor='purple', linewidth=3))

for i, idx in enumerate(pixel_similar_indices):
    plt.subplot(2, 7, i+9)
    plt.imshow(X_test[idx], cmap="binary")
    
    item_label = labels[y_test[idx]]
    
    is_same_category = y_test[idx] == y_test[search_image_idx]
    border_color = 'green' if is_same_category else 'orange'
    title_color = 'green' if is_same_category else 'darkorange'
    
    plt.title(f"{item_label}", fontsize=8, color=title_color)
    plt.axis("off")
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                       edgecolor=border_color, linewidth=2))

plt.suptitle("Búsqueda Visual por Similitud de Imagen\n" +
             "Arriba: Características de Red Neuronal | Abajo: Similitud de Píxeles", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Mostrar estadísticas comparativas
print("=" * 70)
print("COMPARACIÓN DE MÉTODOS DE BÚSQUEDA")
print("=" * 70)

print("\n📊 Método 1 - Características de Red Neuronal (Recomendado):")
same_cat_count = sum([y_test[idx] == y_test[search_image_idx] for idx in similar_indices])
print(f"   Productos de la misma categoría: {same_cat_count}/6 ({same_cat_count/6*100:.1f}%)")
print(f"   Ventaja: Captura patrones semánticos de alto nivel")

print("\n📊 Método 2 - Similitud Directa de Píxeles:")
pixel_same_cat_count = sum([y_test[idx] == y_test[search_image_idx] for idx in pixel_similar_indices])
print(f"   Productos de la misma categoría: {pixel_same_cat_count}/6 ({pixel_same_cat_count/6*100:.1f}%)")
print(f"   Ventaja: Más simple, busca por apariencia visual directa")

print("\n💡 Interpretación:")
if same_cat_count >= pixel_same_cat_count:
    print("   La búsqueda por características de red neuronal es más efectiva")
    print("   porque entiende el 'significado' de la imagen, no solo los píxeles")
else:
    print("   La búsqueda por píxeles funciona bien para este caso")
    print("   pero puede fallar con variaciones de iluminación o pose")

print("\n🔍 Resultados detallados (Método por Red Neuronal):")
for i, idx in enumerate(similar_indices, 1):
    item_label = labels[y_test[idx]]
    similarity_score = similarities[i-1] * 100
    match = "✓" if y_test[idx] == y_test[search_image_idx] else "✗"
    print(f"   {i}. {item_label:15s} - Similitud: {similarity_score:5.1f}% {match}")

# -----------------------------------------------------------------------------
# APLICACIÓN 2: Control de Calidad Automático
# -----------------------------------------------------------------------------
print("\n--- APLICACIÓN 2: Control de Calidad Automático ---")
print("Detectar posibles defectos o clasificaciones incorrectas\n")

# Simular control de calidad: encontrar predicciones con baja confianza
all_predictions = model.predict(X_test_norm, verbose=0)
confidence_scores = np.max(all_predictions, axis=1)

# Umbral de confianza para alertas de calidad
quality_threshold = 0.85
low_confidence_idx = np.where(confidence_scores < quality_threshold)[0][:8]

if len(low_confidence_idx) > 0:
    plt.figure(figsize=(16, 4))
    
    for i, idx in enumerate(low_confidence_idx):
        plt.subplot(2, 4, i+1)
        plt.imshow(X_test[idx], cmap="binary")
        
        predicted_label = labels[np.argmax(all_predictions[idx])]
        true_label = labels[y_test[idx]]
        confidence = confidence_scores[idx]
        
        color = 'orange' if confidence < 0.7 else 'yellow'
        plt.title(f"Confianza: {confidence:.2%}\nReal: {true_label}\nPred: {predicted_label}", 
                  fontsize=8, color=color)
        plt.axis("off")
        
        if confidence < 0.7:
            plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                               edgecolor='red', linewidth=2))
    
    plt.suptitle("⚠️ Control de Calidad: Productos Requieren Inspección Manual", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"⚠️ Alerta: {len(low_confidence_idx)} productos requieren revisión manual")
    print(f"   Confianza promedio: {np.mean(confidence_scores[low_confidence_idx]):.2%}")
else:
    print("✓ Todos los productos cumplen con el estándar de calidad")

# -----------------------------------------------------------------------------
# APLICACIÓN 3: Gestión de Inventario Inteligente
# -----------------------------------------------------------------------------
print("\n--- APLICACIÓN 3: Gestión de Inventario ---")
print("Clasificar y contar automáticamente productos en almacén\n")

# Simular un lote de productos para clasificar
batch_size = 100
batch_indices = np.random.choice(len(X_test), batch_size, replace=False)
batch_images = X_test_norm[batch_indices]
batch_labels = y_test[batch_indices]

# Clasificar el lote completo
batch_predictions = model.predict(batch_images, verbose=0)
predicted_categories = np.argmax(batch_predictions, axis=1)

# Contar productos por categoría
category_counts = {}
for i, category in enumerate(labels):
    count = np.sum(predicted_categories == i)
    category_counts[category] = count

# Visualizar distribución del inventario
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
categories = list(category_counts.keys())
counts = list(category_counts.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

bars = plt.bar(range(len(categories)), counts, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Categoría de Producto', fontweight='bold')
plt.ylabel('Cantidad en Inventario', fontweight='bold')
plt.title('Distribución de Inventario por Categoría', fontweight='bold')
plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.subplot(1, 2, 2)
plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
plt.title('Porcentaje del Inventario Total', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"✓ Clasificados {batch_size} productos automáticamente:")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / batch_size) * 100
    bar = '█' * int(percentage / 2)
    print(f"  {category:15s}: {count:3d} unidades [{bar:20s}] {percentage:5.1f}%")

# -----------------------------------------------------------------------------
# APLICACIÓN 4: Recomendación de Outfits
# -----------------------------------------------------------------------------
print("\n--- APLICACIÓN 4: Recomendación de Outfits ---")
print("Sugerir combinaciones de prendas complementarias\n")

# Definir reglas de combinación de outfits
outfit_rules = {
    'T-shirt/top': ['Trouser', 'Sneaker', 'Bag'],
    'Shirt': ['Trouser', 'Sneaker', 'Bag'],
    'Dress': ['Sandal', 'Bag', 'Coat'],
    'Pullover': ['Trouser', 'Sneaker', 'Bag'],
    'Coat': ['Shirt', 'Trouser', 'Ankle boot']
}

# Seleccionar una prenda base
base_item_idx = 15
base_image = X_test_norm[base_item_idx]
base_label = labels[y_test[base_item_idx]]

print(f"Prenda seleccionada: {base_label}")

# Generar recomendaciones
if base_label in outfit_rules:
    recommended_categories = outfit_rules[base_label]
    print(f"Recomendaciones: {', '.join(recommended_categories)}\n")
    
    plt.figure(figsize=(15, 4))
    
    # Mostrar prenda base
    plt.subplot(1, 4, 1)
    plt.imshow(X_test[base_item_idx], cmap="binary")
    plt.title(f"PRENDA BASE\n{base_label}", fontweight='bold', color='green', fontsize=11)
    plt.axis("off")
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, 
                                       edgecolor='green', linewidth=3))
    
    # Buscar y mostrar prendas complementarias
    for i, rec_category in enumerate(recommended_categories[:3]):
        # Encontrar índice de la categoría recomendada
        cat_idx = labels.index(rec_category)
        # Buscar un producto de esa categoría
        matching_indices = np.where(y_test == cat_idx)[0]
        
        if len(matching_indices) > 0:
            rec_idx = matching_indices[0]
            plt.subplot(1, 4, i+2)
            plt.imshow(X_test[rec_idx], cmap="binary")
            plt.title(f"Recomendación {i+1}\n{rec_category}", fontsize=10)
            plt.axis("off")
    
    plt.suptitle("💡 Sistema de Recomendación de Outfits", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Outfit completo sugerido con {len(recommended_categories)} prendas complementarias")
else:
    print(f"⚠️ No hay reglas de recomendación para {base_label}")

print("\n" + "=" * 70)
print("TAREA COMPLETADA EXITOSAMENTE")
print("=" * 70)