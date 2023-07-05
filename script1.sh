#GRABAR CON MATRIX CREATOR Y GUARDAR LOS AUDIOS EN EL ORDENADOR EN UN .WAV POR PASAJERO
#!/bin/bash
#ejecutar en la terminal

contador=1
#bucle para grabar un número de veces
while [ "$contador" -le 5 ]; do
    #conexión con la raspberry
    sshpass -p raspberry ssh pi@192.168.2.102 << EOF
        cd ~/Documents/CIC
        #grabación el número de segundos que se ponga en el sleep (-1.5 aprox para que se inicie el odaslive)
        (~/odas/bin/odaslive -vc ~/odas/config/odaslive/matrix_creator.cfg) & sleep 10; kill $!
        #borrar los procesos, liberar el espacio
        pid=\$(pgrep odaslive)
        kill -9 \$pid
EOF
    #mandar al ordenador los .raw
    sshpass -p raspberry scp pi@192.168.2.102:/home/pi/Documents/CIC/separated.raw /home/proyectoml/Documentos/Hubert2/Audios/raw/p${contador}.raw
    #pasar de .raw a un .wav por cada uno de los 4 canales
    for ((i=1; i<5; i++)); do
        sox -r 16000 -e signed -b 16 -c 4 Audios/raw/p${contador}.raw Audios/wav/p${contador}.${i}.wav remix $i
    done
    
    contador=$((contador + 1))
done






