# Comparación de modelos Hugging Face para sentimientos en español

Proyecto académico de Tecmilenio, Módulo 2, Semana 5 (Temas 9 y 10). Comparo BERT, DistilBERT y RoBERTuito para clasificar comentarios en español como positivos, neutrales o negativos.

## Caso de uso y datos

La fuente es [`clapAI/MultiLingualSentiment`](https://huggingface.co/datasets/clapAI/MultiLingualSentiment), partición `test`. Seleccioné 3,000 registros declarados en español: 1,000 por clase. Se filtran vacíos y duplicados normalizados. Los campos son `text` (comentario), `label` (sentimiento), `language` (idioma declarado), `source` (fuente) y `domain` (dominio). Ver `data/README.md`.

## Entorno de ejecución

Google Colab, GPU NVIDIA Tesla T4, Python, PyTorch con CUDA, Transformers, Datasets y scikit-learn. Se ejecutó inferencia por lotes de 4 y `max_length=128`. No se midió costo monetario. Las versiones exactas del entorno original no quedaron registradas: `requirements.txt` usa rangos compatibles como punto de partida, no un lockfile.

## Pasos de ejecución

1. Abrir Google Colab y activar GPU T4. Verificar `!nvidia-smi` y `torch.cuda.is_available()`.
2. Instalar dependencias con `%pip install -r requirements.txt` (no reemplazar PyTorch CUDA por una versión solo CPU).
3. Ejecutar `python src/extraer_dataset.py` para crear `data/comentarios_espanol_3000.csv`, o colocar allí el CSV original de Colab.
4. Ejecutar `python src/validar_dataset.py`.
5. Ejecutar `python src/evaluar_modelos.py` con GPU. Se generan métricas y predicciones en `results/`.
6. Abrir `notebooks/comparacion_modelos.ipynb` para una ejecución guiada en Colab.

No se requiere token de Hugging Face para los recursos públicos utilizados. Nunca subir tokens o secretos al repositorio.

## Modelos

- BERT: `finiteautomata/beto-sentiment-analysis`
- DistilBERT: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- RoBERTa: `pysentimiento/robertuito-sentiment-analysis`

Las etiquetas de salida se normalizan a `positive`, `neutral` y `negative` antes de calcular métricas macro.

## Resultados observados

| Modelo | Accuracy | F1 macro | Latencia (ms/texto) |
|---|---:|---:|---:|
| BERT | 63.17 % | 61.58 % | 4.83 |
| DistilBERT | 55.07 % | 48.18 % | 2.81 |
| RoBERTa | **65.80 %** | **63.70 %** | 4.52 |

Ver `results/README.md` para el detalle. Estos resultados corresponden a la ejecución realizada y no a una garantía de rendimiento general.

## Conclusiones

RoBERTa obtuvo la mejor calidad predictiva y una latencia menor que BERT, mientras DistilBERT fue el más rápido pero perdió calidad. Recomiendo RoBERTa **solo como candidato para continuar el prototipo**: ningún modelo alcanzó el umbral académico de 80 % de accuracy. Además, RoBERTa reconoció solo 334 de 1,000 comentarios neutrales. Antes de utilizarlo en producción hace falta revisar etiquetas, dominio, posibles solapamientos con datos de entrenamiento, licencias y la necesidad de fine-tuning.

## Estructura

- `data/`: dataset y descripción de campos.
- `notebooks/`: cuaderno ejecutable.
- `src/`: extracción, validación y evaluación.
- `results/`: resultados y predicciones generadas.
- `requirements.txt`: dependencias.

## Reproducibilidad y limitaciones

La muestra se mezcla con semilla 42, pero para garantizar los mismos registros en el futuro habría que fijar el commit/revisión del dataset. El idioma `es` proviene de metadatos, no de un detector independiente. Los tiempos se midieron una vez, por lotes, sin incluir descarga ni carga inicial. Las licencias de cada checkpoint deben verificarse en sus fichas antes de redistribución o uso comercial.

## Referencias

- Tecmilenio. Tema 9: Selección de modelos preentrenados.
- Tecmilenio. Tema 10: Prototipado rápido y documentación técnica.
- https://huggingface.co/datasets/clapAI/MultiLingualSentiment
