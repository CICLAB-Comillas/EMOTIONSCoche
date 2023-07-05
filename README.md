# EMOTIONSCoche

Podemos entrenarlo o USAR NUESTRO MODELO (no esta perfecto pero es útil para pruebas)

## Pasos para entrenar el modelo

- Asegurarnos que hemos incluido todas las dependencias del archivo requierements.txt
- Descargarnos la base de datos deseada, el siguiente código esta adaptado para la base de CREMAD,  
    - descargar el archive.zip (buscar en internet)
    - extraer los datos en la misma carpeta antes de ejecutar el script
- HubertCrema.py: los parámetros más relevantes son los del training arguments
~~~
        training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        evaluation_strategy="steps",
        num_train_epochs=0.1,
        fp16=False,
        save_steps=10,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-4,
        save_total_limit=2,
        do_train=True,
        do_eval=True,
        do_predict=True
    )
~~~

## Usar nuestro checkpoint

Tras entrenar el modelo, obtuvimos una aproximación. Esta es la alamcenada en checkpoint_28_6
- El modelo esta alamcenado en .bin
- El training_state es un json que va índicando el loss por step (útil para ver gráficas).
- El config también es útil para no tener que importarlo de fuera.
- OjO: hay un sesgo a fear --> si hay silencio lo detecta como fear. 

## Usar todo con el micro

### Raspberry

- Encender la raspberry 
- Conectarla a una wifi sino lo está desde una pantalla

### PC 

- Crear la carpeta Audios con  dos subcarpertas: raw y wav --> se adjuntan unas muestras de ejemplo
- Crear la carpeta Fotos
- Crear un envoriment con todas las dependencias del archivo **requirements.txt**

#### Script.sh

##### Para ejecutarlo:

- chmod +x script1.sh
- ./script1.sh

##### Caracterśiticas más relevantes:

- Se graba ejecutando el **script.sh** desde la terminal (está pensado para Linux-Ubuntu)
    - Se conecta a la raspeberry con: 
        ~~~
         sshpass -p raspberry ssh pi@192.168.2.102
        ~~~
         + raspberry: contraseña de la raspberry
         + 192.168.2.102: dirección IP de la raspeberry --> solo funciona si nuestro ordenador también esta en esta wifi.
    - Hemos establecido que haga 5 grabaciones, se puede modificar en la siguiente línea: 
     ~~~
        while [ "$contador" -le 5 ]; do
     ~~~
    - La duración de cada audio son 10 segundos, se puede reducir(mínimo 3 sec) o ampliar en la linea 12:
        ~~~
        sleep 10;
        ~~~
    - Almacena los 5 .raw en la ruta especificada (cambiar en el sistema deseado) en la línea 18:
        ~~~
         home/proyectoml/Documentos/Hubert2/Audios/raw/p${contador}.raw
        ~~~
        + El contador indica en que grabación estamos
    - La raspberry genera un único .raw y como hay 4 voces en el coche (el micrófono separa la voz de 4 pasjeros), se generan 4 .wav con la voz de cada pasejero (piloto,copiloto...) y se guardan en la carpeta .wav (o la indicada)
    ~~~
         sox -r 16000 -e signed -b 16 -c 4 Audios/raw/p${contador}.raw Audios/wav/p${contador}.${i}.wav remix $i
    ~~~   
#### script2.py

- Podemos inicializar el cálculo de la emoción a la vez con **script2.py**, se pueden ejecutar en paralelo
    + Recorre cada audio de la carpeta .wav por ORDEN: 
        - si no existe el audio1, no pasa al audio2 --> entra en bucle infinito
        - OjO con el nombre de los audios y rutas --> si lo cambias en el archivo de antes, aquí también
    + Genera una imagen por cada 4 .wavs (una grabación) y los mete en la carpeta fotos