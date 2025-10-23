# 🧠 Proyecto de Machine Learning: Clasificación de Ropa y Regresión Lineal

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Caso Práctico](#-caso-práctico)
- [Objetivos](#-objetivos)
- [Requisitos](#-requisitos)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Operación 01: Red Neuronal](#-operación-01-red-neuronal-con-fashion-mnist)
- [Operación 02: Regresión Lineal](#-operación-02-regresión-lineal-simple)
- [Resultados Esperados](#-resultados-esperados)
- [Preguntas y Respuestas](#-preguntas-y-respuestas)
- [Conclusiones](#-conclusiones)
- [Mejoras Futuras](#-mejoras-futuras)
- [Autor](#-autor)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa dos modelos de Machine Learning para resolver problemas en la industria de la moda:

1. **Red Neuronal de Clasificación**: Clasificación automática de prendas de ropa usando el dataset Fashion MNIST
2. **Regresión Lineal Simple**: Análisis de la relación entre temperatura y ventas de ropa

### Tecnologías Utilizadas

- **Python 3.8+**
- **TensorFlow 2.x** - Framework de Deep Learning
- **Keras** - API de alto nivel para redes neuronales
- **Scikit-learn** - Biblioteca de Machine Learning
- **NumPy** - Computación numérica
- **Matplotlib** - Visualización de datos

---

## 🏢 Caso Práctico

Una empresa de moda busca implementar un **sistema automatizado de clasificación de ropa** en su almacén. El sistema debe:

- ✅ Identificar automáticamente diferentes tipos de prendas
- ✅ Clasificar basándose en imágenes
- ✅ Organizar el inventario de manera eficiente
- ✅ Mejorar los tiempos de respuesta para reposición

### Categorías a Clasificar

El sistema reconoce **10 categorías** de ropa:

| #  | Categoría       | Descripción               |
|----|-----------------|---------------------------|
| 0  | T-shirt/top     | Camisetas y tops          |
| 1  | Trouser         | Pantalones                |
| 2  | Pullover        | Suéteres                  |
| 3  | Dress           | Vestidos                  |
| 4  | Coat            | Abrigos                   |
| 5  | Sandal          | Sandalias                 |
| 6  | Shirt           | Camisas                   |
| 7  | Sneaker         | Zapatillas deportivas     |
| 8  | Bag             | Bolsos                    |
| 9  | Ankle boot      | Botas de tobillo          |

---

## 🎓 Objetivos

Al concluir este proyecto, serás capaz de:

- [x] Crear y entrenar una red neuronal con Python y TensorFlow
- [x] Implementar y graficar una regresión lineal simple
- [x] Procesar y normalizar datos de imágenes
- [x] Evaluar el rendimiento de modelos de ML
- [x] Visualizar resultados y métricas de evaluación
- [x] Aplicar técnicas de Machine Learning a problemas reales

---

## 💻 Requisitos

### Hardware Recomendado

- **RAM**: 4GB mínimo (8GB recomendado)
- **GPU**: Opcional (acelera el entrenamiento)
- **Almacenamiento**: 1GB libre

### Software Necesario

```
Python >= 3.8
TensorFlow >= 2.0
NumPy >= 1.19
Matplotlib >= 3.3
Scikit-learn >= 0.24
```

---

## 🚀 Instalación y Configuración

### Opción 1: Google Colab (Recomendado - No requiere instalación)

1. Ve a [Google Colab](https://colab.research.google.com)
2. Crea un nuevo notebook
3. Copia y pega el código
4. ¡Ejecuta! Todo está preinstalado ✨

### Opción 2: Instalación Local

```bash
# 1. Clonar o descargar el proyecto
git clone https://github.com/tu-usuario/fashion-ml-project.git
cd fashion-ml-project

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install --upgrade tensorflow numpy matplotlib scikit-learn

# 4. Verificar instalación
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')"
```

---

## 📁 Estructura del Proyecto

```
fashion-ml-project/
│
├── README.md                          # Este archivo
├── main.py                            # Código principal del proyecto
├── requirements.txt                   # Dependencias del proyecto
│
├── data/                              # (Auto-descargado por TensorFlow)
│   └── fashion_mnist/                 # Dataset Fashion MNIST
│
├── models/                            # Modelos guardados (opcional)
│   └── fashion_classifier.h5
│
├── results/                           # Gráficos y resultados
│   ├── training_history.png
│   ├── predictions.png
│   └── regression_plot.png
│
└── docs/                              # Documentación adicional
    ├── architecture.md
    └── analysis.md
```

---

## 🧠 Operación 01: Red Neuronal con Fashion MNIST

### Arquitectura del Modelo

```
┌─────────────────────────────────────┐
│   INPUT: Imagen 28x28 píxeles       │
│         (Escala de grises)          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   FLATTEN LAYER                     │
│   Convierte 28x28 → Vector de 784   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   DENSE LAYER (128 neuronas)        │
│   Activation: ReLU                  │
│   Parámetros: 100,480               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   OUTPUT LAYER (10 neuronas)        │
│   Activation: Softmax               │
│   Parámetros: 1,290                 │
└──────────────┬──────────────────────┘
               │
               ▼
        Clasificación
     (10 categorías)
```

### Dataset Fashion MNIST

![Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

**Características:**
- **60,000** imágenes de entrenamiento
- **10,000** imágenes de prueba
- Tamaño: **28x28 píxeles**
- Escala de grises (1 canal)
- 10 clases balanceadas

### Proceso de Entrenamiento

#### Paso 1: Carga de Datos

```python
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

#### Paso 2: Normalización

Las imágenes se normalizan de [0, 255] a [0, 1]:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

**¿Por qué normalizar?**
- Acelera el entrenamiento
- Mejora la convergencia del modelo
- Estabiliza el gradiente descendente

#### Paso 3: Configuración del Modelo

```python
model.compile(
    optimizer='adam',          # Optimizador adaptativo
    loss='sparse_categorical_crossentropy',  # Para clasificación multi-clase
    metrics=['accuracy']       # Métrica a monitorear
)
```

#### Paso 4: Entrenamiento (30 épocas)

```python
history = model.fit(
    X_train_norm, 
    y_train, 
    epochs=30,                      # 30 iteraciones completas
    validation_data=(X_test_norm, y_test),
    verbose=1
)
```

### Visualizaciones Generadas

#### 1. Muestras del Dataset
Muestra 20 imágenes aleatorias con sus etiquetas para explorar el dataset.

#### 2. Curvas de Aprendizaje
Gráficos que muestran:
- **Precisión** (Accuracy) en entrenamiento y validación
- **Pérdida** (Loss) en entrenamiento y validación

Estos gráficos ayudan a detectar:
- ✅ **Buen ajuste**: Curvas convergentes
- ⚠️ **Overfitting**: Precisión de entrenamiento >> validación
- ⚠️ **Underfitting**: Ambas precisiones bajas

#### 3. Predicciones de Ejemplo
Muestra 10 predicciones con colores:
- 🟢 **Verde**: Predicción correcta
- 🔴 **Rojo**: Predicción incorrecta

---

## 📊 Operación 02: Regresión Lineal Simple

### Objetivo

Analizar la relación entre la **temperatura** y las **ventas** de ropa para predecir el comportamiento de ventas según condiciones climáticas.

### Modelo Matemático

La regresión lineal busca la ecuación:

```
Ventas = β₀ + β₁ × Temperatura
```

Donde:
- **β₀** = Intercepto (ventas base)
- **β₁** = Pendiente (cambio en ventas por grado)

### Dataset Utilizado

```python
Temperatura (°C): [15, 16, 18, 20, 21, 23, 25, 27, 30, 32]
Ventas:          [500, 520, 560, 580, 600, 640, 680, 700, 760, 800]
```

### Proceso de Análisis

#### 1. División de Datos

```
80% Entrenamiento (8 muestras)
20% Prueba (2 muestras)
```

#### 2. Entrenamiento del Modelo

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 3. Evaluación

**Métricas calculadas:**
- **R² (Coeficiente de determinación)**: Mide qué tan bien el modelo explica la variación
- **MSE (Error Cuadrático Medio)**: Promedio de errores al cuadrado
- **RMSE (Raíz del MSE)**: Error en unidades originales

### Interpretación de R²

| Valor de R² | Interpretación                    |
|-------------|-----------------------------------|
| 0.9 - 1.0   | Ajuste excelente ⭐⭐⭐          |
| 0.7 - 0.9   | Ajuste bueno ⭐⭐                |
| 0.5 - 0.7   | Ajuste moderado ⭐              |
| < 0.5       | Ajuste pobre                     |

### Visualizaciones

#### 1. Gráfico de Regresión
- **Puntos azules**: Datos reales
- **Línea roja**: Modelo ajustado
- **Puntos verdes**: Datos de prueba

#### 2. Análisis de Residuos
- **Gráfico de dispersión**: Detecta patrones en errores
- **Histograma**: Distribución de errores (idealmente normal)

---

## 📈 Resultados Esperados

### Red Neuronal

| Métrica              | Valor Esperado |
|----------------------|----------------|
| Precisión Entrenamiento | 88-92%      |
| Precisión Validación    | 86-90%      |
| Pérdida Final           | 0.25-0.35   |
| Tiempo por Época        | 2-5 segundos|

### Regresión Lineal

| Métrica              | Valor Esperado |
|----------------------|----------------|
| R²                   | > 0.95         |
| RMSE                 | < 30           |
| Correlación          | Positiva fuerte|

---

## ❓ Preguntas y Respuestas

### 1. ¿Qué función cumple la capa Flatten en el modelo de red neuronal?

**Respuesta:**

La capa **Flatten** transforma las imágenes bidimensionales de 28×28 píxeles en un vector unidimensional de 784 valores. Esta conversión es necesaria porque las capas Dense solo procesan datos en formato vectorial.

**Ejemplo visual:**
```
Imagen 28x28          Flatten          Vector 784
┌────────┐              →            [p₁, p₂, ..., p₇₈₄]
│░░▓▓░░░░│
│░▓▓▓▓░░░│
│░░▓▓░░░░│
└────────┘
```

---

### 2. ¿Qué descubriste sobre la relación entre temperatura y ventas?

**Respuesta:**

Se descubrió una **relación lineal positiva muy fuerte** entre temperatura y ventas:

- **Correlación**: A mayor temperatura → mayores ventas
- **R² ≈ 0.95-0.99**: La temperatura explica casi toda la variación en ventas
- **Interpretación práctica**: Por cada grado que sube la temperatura, las ventas aumentan aproximadamente ~20 unidades
- **Implicación de negocio**: Se pueden predecir las ventas usando pronósticos meteorológicos

---

### 3. ¿Qué mejoras le harías a la red neuronal para aumentar la precisión?

**Respuestas:**

#### Mejoras de Arquitectura:
1. **Redes Convolucionales (CNN)**
   ```python
   model = keras.Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
       MaxPooling2D((2,2)),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D((2,2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

2. **Más Capas Ocultas (Deep Learning)**
   ```python
   Dense(256, activation='relu'),
   Dense(128, activation='relu'),
   Dense(64, activation='relu'),
   ```

3. **Dropout para evitar Overfitting**
   ```python
   Dense(128, activation='relu'),
   Dropout(0.3),  # Desactiva 30% de neuronas
   ```

#### Mejoras de Datos:
4. **Data Augmentation**
   - Rotación aleatoria
   - Zoom
   - Desplazamiento horizontal

5. **Batch Normalization**
   - Normaliza activaciones entre capas
   - Acelera el entrenamiento

---

### 4. ¿Cómo puede mejorar la precisión al aumentar el número de épocas?

**Respuesta:**

**Ventajas de más épocas:**
- ✅ Más iteraciones de aprendizaje
- ✅ Ajuste más fino de pesos
- ✅ Mejor convergencia del modelo

**Gráfico conceptual:**

```
Precisión
    ^
    |     ┌────────── Plateau (óptimo)
    |    ╱
    |   ╱         ⚠️ Overfitting
    |  ╱         ╱
    | ╱         ╱
    |╱_________╱
    └──────────────────────> Épocas
    5   10   15   20   30
```

**Consideraciones importantes:**
- ⚠️ **Overfitting**: Demasiadas épocas → memorización
- ✅ **Early Stopping**: Detener cuando validación deja de mejorar
- 📊 **Monitoreo**: Observar gráficas de precisión/pérdida

**Número óptimo:**
- Fashion MNIST: 20-40 épocas típicamente
- Usar validación para decidir

---

### 5. ¿Qué otras aplicaciones prácticas para esta red neuronal en la industria de la moda?

**Respuestas:**

#### 🏪 Retail y E-commerce
- **Búsqueda visual**: "Encuentra productos similares a esta foto"
- **Probadores virtuales**: Identificar tipo de prenda para AR
- **Recomendación de outfits**: Sugerir combinaciones automáticas

#### 🏭 Producción y Logística
- **Control de calidad**: Detectar defectos en prendas
- **Clasificación automática**: Organizar inventario en almacenes
- **Gestión de devoluciones**: Categorizar productos devueltos

#### 📱 Marketing y Análisis
- **Análisis de tendencias**: Identificar estilos populares en redes sociales
- **Detección de falsificaciones**: Verificar autenticidad de productos
- **Segmentación de catálogos**: Organizar automáticamente colecciones

#### ♻️ Sostenibilidad
- **Reciclaje textil**: Clasificar ropa para programas de reciclaje
- **Mercado de segunda mano**: Categorizar prendas usadas
- **Donaciones**: Organizar ropa para donación

#### 🤖 Experiencia de Cliente
- **Asistente virtual de estilo**: "¿Qué prenda es esta?"
- **Organización de guardarropa**: Apps para gestión de armario personal
- **Alertas de stock**: Notificar cuando productos similares están disponibles

---

## 💡 Conclusiones

### Red Neuronal (Fashion MNIST)

✅ **Éxitos:**
- Modelo simple pero efectivo (85-90% precisión)
- Entrenamiento rápido (pocos minutos)
- Aplicable a problemas reales de clasificación

📝 **Aprendizajes:**
- La normalización es crucial para el rendimiento
- 30 épocas son suficientes para este dataset
- La arquitectura simple funciona bien para Fashion MNIST

### Regresión Lineal

✅ **Éxitos:**
- Relación clara entre temperatura y ventas
- Modelo interpretable y explicable
- Útil para planificación de inventario

📝 **Aprendizajes:**
- Las relaciones lineales son comunes en negocios
- R² alto indica buen poder predictivo
- Análisis de residuos valida la calidad del modelo

---

## 🚀 Mejoras Futuras

### Para la Red Neuronal

- [ ] Implementar CNN (Redes Convolucionales)
- [ ] Agregar Dropout y Batch Normalization
- [ ] Probar con Transfer Learning (VGG16, ResNet)
- [ ] Implementar data augmentation
- [ ] Crear API REST para clasificación en producción
- [ ] Optimizar para deployment en móviles (TensorFlow Lite)

### Para la Regresión

- [ ] Incluir más variables (humedad, día de semana, promociones)
- [ ] Probar regresión polinomial
- [ ] Implementar series temporales (ARIMA, LSTM)
- [ ] Crear dashboard interactivo para predicciones
- [ ] Integrar con datos reales de punto de venta

---

## 📚 Referencias y Recursos

### Documentación Oficial
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Datasets
- [Fashion MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)
- [Original MNIST](http://yann.lecun.com/exdb/mnist/)

### Tutoriales Recomendados
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### Artículos Académicos
- Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms
- Deep Learning for Computer Vision

---

## 👨‍💻 Autor

**Tu Nombre**
- 📧 Email: jhordangonzalo234@gmail.com
- 💻 GitHub: [@tu-usuario](https://github.com/Jhordan234)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Dataset Fashion MNIST por Zalando Research
- Comunidad de TensorFlow y Keras
- Instructores y compañeros del curso

---

## 📞 Contacto y Soporte

Si tienes preguntas o necesitas ayuda:

1. 📧 **Email**: jhordangonzalo234@gmail.com
2. 🐛 **Issues**: Reporta bugs en la sección de Issues
3. 📖 **Wiki**: Consulta la wiki del proyecto para más detalles

---

<div align="center">

### ⭐ Si este proyecto te fue útil, considera darle una estrella

**Hecho con ❤️ y 🧠 usando Python y TensorFlow**

![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-FF6F00.svg)

</div>