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

print("\n" + "=" * 70)
print("TAREA COMPLETADA EXITOSAMENTE")
print("=" * 70)