# Run in Docker Guide :  tensorflow-gpu 2.0 + cuda + jupyter for Deep Learning use

## Agrega el usuario al grupo docker para evitar correr con sudo

    sudo usermod -aG docker $USER

### Requiriments in Host: 
    -Nvidia Drivers
    -docker
    -nvidia-docker2
    -docker-compose > 1.19

### Main Host Directory structure :

-docker : docker-compose file and saved images

-Workspace : volumen mapeado del host al contenedor

    src :  todos los archivos fuentes del proyecto

    datasets : COCO dataset , imagenes y annotations.

    pickle_saves : guardado de codificaciones de imagenes y de texto, imagenes preprocesadas con inception v3 y tokenizer .

    checkpoints :   guardado de entrenamiendo de modelos

## installation of docker-compose 1.27.1 : 
    sudo curl -L "https://github.com/docker/compose/releases/download/1.27.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

## start container (./docker/)
    docker-compose up -d

## enter to container
    docker exec -ti tf_container /bin/bash

## execute python code from container (/workspace/src/)
    python code.py

## remove containers (./docker/)
    docker-compose down 
