<img src="https://drive.google.com/uc?export=view&id=1fBDbTvato9MGgCJ-yMGF_cw-f8K1AZ3K" title="Facultad de Ciencias Exactas Físicas y Naturales" width="200" img align="right"/>

## Practica Profesional Supervisada

#### Nuevo Código Fuente modularizado

- Captioner.py : Codigo fuente relacionado al modelo del image captioner (instanciamiento,caching de imagenes,pre-procesamiento de imagenes,cargado de checkpoints,entrenamiento y generación de los embeddings de las imagenes y evaluación con salida en consola, con una imagen específica o un grupo)
- Dataset.py : Codigo fuente relacionado al manejo de los datasets nescesarios para el posterior entrenamiento de los modelos (descarga , tokenizacion de las captions ,procesamiento con IncV3 de las imagenes y creacion de los datasets)
- Text_encoder.py : Codigo fuente relacionado al modelo del codificador de texto (instanciamiento,cargado de checkpoints,evaluacion con una caption específica,generacion de los embeddings de las captions y entrenamiento)
- Evalutaion.py : Codigo fuente relacionado a la evaluacion de los modleos (Calculo recall,generacion de archivos tsv para la visualizacion del espacio tensorial de los embeddings)