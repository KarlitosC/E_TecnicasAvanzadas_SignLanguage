# TLY1102 - Evaluación 2: CNN para Sign Language MNIST

Proyecto de Técnicas Avanzadas de Machine Learning. Se implementó una Red Neuronal Convolucional (CNN) para clasificar las señas de las vocales **A, E, I, O y U** de Sign Language MNIST, y se compararon sus resultados con el MLP desarrollado en la Evaluación 1.

## Entregables

- [Notebook de la E2](notebooks/E2_CNN/TLY1102_E2_CNN_SignLanguage.ipynb)
- [Informe técnico en Markdown](reports/Informe_Tecnico_E2_CNN.md)
- Presentación: se agregará en `presentation/` cuando esté lista.

## Estructura principal

- `notebooks/E1_MLP/`: notebook de la Evaluación 1.
- `notebooks/E2_CNN/`: notebook ejecutable de la CNN.
- `reports/`: informe técnico y matriz de responsabilidades.
- `presentation/`: presentación del proyecto, pendiente de incorporar.

## Datos y ejecución

Los archivos `sign_mnist_train.csv` y `sign_mnist_test.csv` se obtienen del material del curso y no están incluidos en el repositorio por su tamaño.

Para ejecutar el notebook en Google Colab:

1. Abre el notebook de la E2.
2. En el panel **Archivos**, sube manualmente ambos CSV.
3. Ejecuta las celdas en orden con **Entorno de ejecución → Ejecutar todo**.

Los archivos quedan en la sesión temporal de Colab. Si la sesión se reinicia, tendrás que subirlos nuevamente.

## Resultados de la comparación

| Métrica en prueba | MLP (E1) | CNN (E2) |
|---|---:|---:|
| Accuracy | 93,43 % | 96,93 % |
| Precision macro | 93,74 % | 97,26 % |
| Recall macro | 91,94 % | 95,93 % |
| F1-score macro | 92,26 % | 96,33 % |

La CNN obtuvo mejores métricas de prueba en esta ejecución, con más parámetros y mayor tiempo de entrenamiento. El informe y el notebook incluyen el análisis y los resultados detallados.

## Equipo

Trabajo realizado por un único integrante. La matriz de responsabilidades está incluida en el informe técnico.
