# Talleres de las semanas 8 y 9

Cuatro actividades del [repositorio guía de EAFIT SI3003](https://github.com/EAFIT-IA/si3003-artificial-intelligence/tree/dfbcd874eb8bc27a3f3c258b0e7fdc41a6d9dbd3), basadas en el commit `dfbcd874eb8bc27a3f3c258b0e7fdc41a6d9dbd3`.

Se conservan los nombres, las rutas, las consignas y el orden de las celdas originales; las soluciones están integradas en el código y en respuestas añadidas. La autoría del material guía corresponde a sus autores. Cada notebook documenta los cambios al final.

## Organización

| Semana | Notebook original | Actividad |
|---|---|---|
| 8 | [taller_transfer_learning_datos_propios_keras3.ipynb](lecture8/transfer_learning/taller_transfer_learning_datos_propios_keras3.ipynb) | Clasificador de tres clases, EfficientNetB0, split 70/15/15, entrenamiento, fine-tuning, métricas y errores. |
| 8 | [Lecture_08_Yolo_Intro.ipynb](lecture8/yolo/Lecture_08_Yolo_Intro.ipynb) | Detección de objetos y aplicación a una imagen propia. |
| 9 | [05_ejercicio_nvidia_documentos.ipynb](Lecture9/05_ejercicio_nvidia_documentos.ipynb) | Asistente sobre un PDF con NVIDIA, Gradio, historial y evaluación de respuestas. |
| 9 | [tokenization_Introduction.ipynb](Lecture9/tokenization_Introduction.ipynb) | Word2Vec, vecinos de un animal, analogías y proyección PCA 3D. |

Se mantiene la capitalización del repositorio guía: `lecture8` y `Lecture9`. No se duplican los otros ejercicios del repositorio ni se añaden los notebooks de ejemplo adicionales, datasets, pesos o ZIPs.

## Estado real

**Implementación preparada y validación estática/offline realizada; ejecución empírica pendiente.** Los notebooks no contienen resultados de entrenamiento, detecciones ni respuestas de API inventados. Sus salidas y contadores están vacíos.

Se verificaron formato Jupyter, sintaxis y correspondencia con las celdas originales. Las pruebas auxiliares utilizan datos sintéticos o clientes simulados: no demuestran el desempeño de los modelos reales.

La guía de transfer learning exige el notebook ejecutado. Antes de entregar al docente es necesario ejecutar con datos reales, revisar los resultados y guardar las salidas. Los enunciados `TODO` del profesor permanecen intactos; los parámetros de código se han resuelto.

## Ejecutar

1. Descarga el `.ipynb` que corresponda. Ábrelo en Jupyter local o en [Google Colab](https://colab.research.google.com/) mediante **Archivo → Subir notebook**.
2. Lee la configuración inicial y activa la instalación opcional si faltan dependencias. Reinicia el kernel cuando se indique.
3. Configura los datos y activa explícitamente las descargas, entrenamiento o servicios que quieras utilizar. Con los indicadores desactivados no se generan resultados reales.
4. Ejecuta en orden; para transfer learning es recomendable una GPU. Word2Vec necesita principalmente RAM.
5. Completa las observaciones basadas en las salidas reales y guarda el notebook ejecutado.

Si un botón **Open in Colab** del texto original apunta al repositorio del profesor, abrirá su guía, no esta solución. Sube el archivo descargado de este repositorio para trabajar con la versión completada. Cada notebook es independiente y no requiere los otros ejemplos de la guía.

## Qué falta por actividad

### Semana 8 — Transfer learning

- Propuesta: bicicleta, motocicleta y autobús. Puedes cambiarla antes de recolectar.
- Recolecta y revisa aproximadamente 100–150 imágenes útiles por clase; comprueba licencias antes de redistribuirlas.
- Registra exclusiones y la observación de limpieza. Marca `DATASET_REVISADO=True` únicamente cuando hayas inspeccionado el dataset.
- Ejecuta el entrenamiento con base congelada y fine-tuning, y guarda curvas, métricas, matrices de confusión y comparación.
- Completa el análisis visual de al menos cinco errores si existen. Si hay menos, documenta la limitación sin fabricar fallos.
- La limpieza y el split conservan los originales. EfficientNet recibe píxeles [0,255]; BatchNorm permanece congelada. La selección se basa en validation, no en test.

### Semana 8 — YOLO

- Aporta una imagen propia y activa la inferencia.
- Guarda las detecciones, la comparación de umbrales y una observación visual de aciertos, omisiones o confusiones.
- Sin anotaciones de referencia no se calculan accuracy, recall ni mAP: el número de cajas no es una medida de aciertos.

### Semana 9 — Asistente documental

- Aporta el PDF o utiliza la descarga opcional indicada. El lector no aplica OCR a un PDF escaneado.
- Configura `NVIDIA_API_KEY` mediante el entorno, secretos de Colab o entrada oculta; nunca la pegues en el código ni la imprimas.
- Activa el servicio y ejecuta las cinco preguntas del ejercicio y, opcionalmente, la pregunta de control adicional.
- Revisa evidencia y páginas para cada respuesta. Las respuestas reales y su valoración siguen pendientes.
- El proveedor recibe el documento y las preguntas; no uses información confidencial sin autorización. El modelo requiere acceso y cuota en tu cuenta.

### Semana 9 — Word2Vec

- Carga `word2vec-google-news-300` mediante la descarga explícita o un archivo local confiable. Requiere aproximadamente 1.6 GB de descarga y varios GB de RAM.
- Obtén los diez vecinos, ejecuta las analogías y genera los gráficos PCA 3D.
- Contrasta los resultados con las hipótesis e interpreta los términos sorprendentes y las agrupaciones de PCA. El resumen automático no sustituye esa interpretación humana. La alternativa GloVe está identificada como experimento diferente; no sustituye el modelo de la guía.

No publiques credenciales, imágenes sin permiso ni datos privados al guardar y subir las salidas.
