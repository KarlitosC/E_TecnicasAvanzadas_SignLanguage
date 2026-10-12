# Informe técnico: Clasificación de señas mediante una CNN

**Asignatura:** Técnicas Avanzadas de Machine Learning (TLY1102)  
**Evaluación:** Evaluación 2  
**Estudiante(s):** Carlos Navarrete
**Fecha:** 11/10/2026

## 1. Resumen

Este trabajo implementa una red neuronal convolucional (CNN) para clasificar imágenes de señas correspondientes a las vocales **A, E, I, O y U** del conjunto Sign Language MNIST. Se conserva la variante utilizada en la Evaluación 1, donde se clasificaron esas mismas cinco vocales con un Perceptrón Multicapa (MLP). Esto permite comparar los dos modelos sobre la misma tarea y el mismo conjunto de prueba.

La CNN obtuvo **96,93 % de accuracy** y **96,33 % de F1-score macro** en prueba. Superó las métricas de prueba del MLP, aunque utilizó aproximadamente el doble de parámetros y tardó más en entrenarse. De las 1.629 imágenes de prueba, clasificó correctamente 1.579; la mayor parte de los errores correspondió a imágenes de la vocal O.

## 2. Problema y variante

El problema consiste en reconocer la vocal representada en una imagen de una mano que realiza una seña. La variante de esta evaluación restringe la clasificación a cinco clases: **A, E, I, O y U**. Las etiquetas originales del conjunto se filtraron y se remapearon a índices consecutivos: A=0, E=1, I=2, O=3 y U=4.

Se mantuvo esta variante de la E1 para que la comparación entre el MLP y la CNN fuera directa. Las imágenes se representan en escala de grises y tienen un tamaño de 28 × 28 píxeles.

## 3. Datos y preparación

Los archivos de Sign Language MNIST contienen una columna de etiqueta y 784 columnas de píxeles. Se conservaron únicamente las filas de las vocales seleccionadas. Cada fila se transformó en una imagen de 28 × 28 píxeles con un canal, y los valores de píxel se normalizaron dividiéndolos por 255 para quedar entre 0 y 1.

| Conjunto | Imágenes | Forma de entrada |
|---|---:|---|
| Entrenamiento | 4.481 | 28 × 28 × 1 |
| Validación | 1.121 | 28 × 28 × 1 |
| Prueba | 1.629 | 28 × 28 × 1 |

La validación se separó desde el conjunto de entrenamiento con una proporción de 20 %, muestreo estratificado y semilla 29. El conjunto de prueba se mantuvo separado para la evaluación final.

La cantidad de imágenes por vocal fue:

| Vocal | Entrenamiento original filtrado | Prueba |
|---|---:|---:|
| A | 1.126 | 331 |
| E | 957 | 498 |
| I | 1.162 | 288 |
| O | 1.196 | 246 |
| U | 1.161 | 266 |

## 4. Diseño de la CNN y decisiones

La arquitectura se implementó con TensorFlow/Keras. Recibe imágenes de forma 28 × 28 × 1 y contiene las siguientes capas:

| Orden | Capa | Configuración | Propósito |
|---:|---|---|---|
| 1 | Conv2D | 32 filtros, 3 × 3, ReLU, `padding="same"` | Detectar patrones locales iniciales, como bordes y trazos. |
| 2 | MaxPooling2D | 2 × 2 | Reducir las dimensiones espaciales conservando activaciones relevantes. |
| 3 | Conv2D | 64 filtros, 3 × 3, ReLU, `padding="same"` | Aprender patrones visuales más complejos. |
| 4 | MaxPooling2D | 2 × 2 | Reducir nuevamente el mapa espacial. |
| 5 | Flatten | — | Convertir los mapas de características en un vector. |
| 6 | Dense | 64 unidades, ReLU | Combinar las características extraídas. |
| 7 | Dense | 5 unidades, softmax | Entregar una probabilidad para cada vocal. |

Las convoluciones se utilizan para aprender características locales conservando la estructura espacial de la imagen. El pooling reduce el tamaño de las representaciones entre convoluciones. La capa softmax produce las cinco salidas de clasificación.

La configuración de entrenamiento fue:

- Optimizador: Adam.
- Función de pérdida: `sparse_categorical_crossentropy`, apropiada para etiquetas enteras de 0 a 4.
- Métrica monitoreada: accuracy.
- Épocas: 10.
- Tamaño de lote: 32.
- Parámetros entrenables: 219.909.

De acuerdo con las instrucciones de la profesora, no se agregaron técnicas ni capas de regularización.

## 5. Entrenamiento

Durante el entrenamiento, la accuracy de validación llegó a 97,41 % en la primera época y a 100 % desde la segunda. La accuracy de entrenamiento llegó a 100 % desde la tercera época. La pérdida de entrenamiento y validación disminuyó a lo largo de las diez épocas.

El tiempo de entrenamiento registrado fue **81,86 segundos**. Las curvas de accuracy y pérdida por época se presentan en el notebook de E2.

La accuracy perfecta observada en entrenamiento y validación no se trasladó completamente al conjunto de prueba. Por ello, el resultado de prueba se considera la referencia principal para describir el desempeño ante datos reservados.

## 6. Evaluación de la CNN

En el conjunto de prueba, compuesto por 1.629 imágenes, la CNN obtuvo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 96,93 % |
| Precision macro | 97,26 % |
| Recall macro | 95,93 % |
| F1-score macro | 96,33 % |
| Aciertos | 1.579 de 1.629 |
| Errores | 50 de 1.629 |

El reporte por clase fue:

| Vocal | Precision | Recall | F1-score | Imágenes |
|---|---:|---:|---:|---:|
| A | 0,95 | 1,00 | 0,97 | 331 |
| E | 0,97 | 1,00 | 0,99 | 498 |
| I | 0,94 | 1,00 | 0,97 | 288 |
| O | 1,00 | 0,80 | 0,89 | 246 |
| U | 1,00 | 1,00 | 1,00 | 266 |

La matriz de confusión —filas: clase real; columnas: clase predicha, en orden A, E, I, O, U— fue:

| Real / predicha | A | E | I | O | U |
|---|---:|---:|---:|---:|---:|
| A | 331 | 0 | 0 | 0 | 0 |
| E | 0 | 498 | 0 | 0 | 0 |
| I | 0 | 0 | 288 | 0 | 0 |
| O | 17 | 15 | 18 | 196 | 0 |
| U | 0 | 0 | 0 | 0 | 266 |

La CNN reconoció correctamente todas las imágenes de prueba de A, E, I y U. En cambio, clasificó correctamente 196 de las 246 imágenes de O; las otras 50 se predijeron como A, E o I. Esta clase concentra los errores observados.

## 7. Comparación con el MLP de la E1

La comparación utiliza las métricas finales registradas en la E1 y E2. Ambos modelos trabajaron con la misma variante del problema, particiones, diez épocas y tamaño de lote 32. El MLP utilizó SGD y la CNN utilizó Adam; por lo tanto, la diferencia observada corresponde a los modelos con sus configuraciones respectivas y no puede atribuirse únicamente a la arquitectura.

| Métrica | MLP (E1) | CNN (E2) |
|---|---:|---:|
| Accuracy en prueba | 93,43 % | 96,93 % |
| Precision macro en prueba | 93,74 % | 97,26 % |
| Recall macro en prueba | 91,94 % | 95,93 % |
| F1-score macro en prueba | 92,26 % | 96,33 % |
| Accuracy final de entrenamiento | 98,01 % | 100,00 % |
| Accuracy final de validación | 98,48 % | 100,00 % |
| Parámetros | 109.061 | 219.909 |
| Tiempo registrado | 6,02 s | 81,86 s |

La CNN superó al MLP en 3,50 puntos porcentuales de accuracy y 4,07 puntos de F1 macro en prueba. También obtuvo mejores valores de precision y recall macro. El MLP fue más rápido y tuvo menos parámetros. En estas ejecuciones, la CNN ofreció mayor desempeño de clasificación con un costo de entrenamiento y tamaño superiores.

El resultado de validación de ambos modelos fue mayor que el de prueba. En particular, la diferencia entre la validación y la prueba de la CNN indica que no se debe usar solo la accuracy de validación para describir su rendimiento final; por eso se reportan las métricas del conjunto de prueba reservado.

## 8. Análisis de predicciones y visualizaciones

### Ejemplos correctos e incorrectos

El notebook presenta cinco ejemplos correctamente clasificados y cinco errores. Los ejemplos incorrectos seleccionados pertenecen a la clase O y fueron predichos como A, E o I, en concordancia con la matriz de confusión. La visualización permite revisar casos concretos junto con sus etiquetas reales y predichas.

### Feature Maps

Se visualizaron ocho mapas de activación de cada una de las dos capas convolucionales para un ejemplo de O mal clasificado. En la primera capa se observan representaciones con mayor detalle espacial. En la segunda, la representación es más pequeña y abstracta; después del segundo pooling tiene una resolución espacial de 7 × 7. Los mapas muestran respuestas distintas de los filtros ante partes de la imagen, aunque por sí solos no identifican de manera semántica cada patrón.

### Mapas de activación por clase (Grad-CAM)

Se generó un Grad-CAM para un ejemplo de prueba de cada vocal. En los cinco ejemplos seleccionados, la predicción coincidió con la clase real. La superposición de colores destaca las zonas que más contribuyeron a la puntuación de la clase objetivo. Esta visualización ayuda a inspeccionar dónde se concentra la activación, pero no debe interpretarse como una explicación causal completa de la decisión.

## 9. Matriz de responsabilidades

Completar esta tabla con los integrantes reales y sus aportes. Si el trabajo es individual, dejar una sola fila con el nombre de quien realizó las tareas. No incluir responsabilidades que no correspondan a la participación efectiva.

| Integrante | Responsabilidades y aportes |
|---|---|
| Carlos Navarrete | Preparación y revisión de los datos; diseño e implementación de la CNN; entrenamiento y evaluación; análisis de errores y visualizaciones; comparación con la E1; elaboración y revisión del informe. |

## 10. Conclusiones

En la variante de clasificación de las vocales A, E, I, O y U, la CNN alcanzó 96,93 % de accuracy y 96,33 % de F1 macro en el conjunto de prueba, superando las métricas registradas por el MLP de la E1. Los mapas de características permiten observar cómo las capas convolucionales transforman las imágenes y Grad-CAM ofrece una visualización de las regiones activadas para ejemplos de cada clase.

La principal dificultad de la CNN fue la vocal O, de la que 50 imágenes se confundieron con A, E o I. En términos de recursos, la CNN tuvo aproximadamente el doble de parámetros que el MLP y tardó más en entrenarse. Por ello, los resultados muestran un mejor desempeño de clasificación en esta ejecución, junto con un costo computacional mayor.

La comparación debe interpretarse considerando que el MLP usó SGD y la CNN Adam. Además, la evaluación respetó la indicación docente de no utilizar técnicas de regularización. Como siguiente mejora experimental, y solo si se solicita en otra instancia, podría compararse bajo configuraciones de optimización controladas; esa comparación no forma parte de los resultados presentados aquí.

## 11. Reproducción del trabajo

El notebook ejecutable de la CNN se encuentra en `notebooks/E2_CNN/`. Para reproducir el trabajo:

1. Disponer de los archivos `sign_mnist_train.csv` y `sign_mnist_test.csv` en el entorno de ejecución o en la ubicación indicada por el notebook.
2. Abrir el notebook de E2 en Google Colab o en un entorno compatible con TensorFlow/Keras.
3. Ejecutar las celdas en orden, desde la configuración y carga de datos hasta las conclusiones.

Los resultados de tiempo pueden variar según el entorno de ejecución. El notebook contiene las salidas registradas en la ejecución presentada.
