import argostranslate.package
import argostranslate.translate
import socket
import os
import os.path
import libretranslatepy as lt
from langdetect import detect
from libretranslatepy import LibreTranslateAPI

# Idiomas para traducción sin conexión
ingles = "en"
español = "es"
# Función para intentar realizar una conexión a internet
def check_internet():
    try:
        # Intenta conectar con un host de confianza
        socket.create_connection(("www.google.com", 80))
        
        return True
    except OSError:
        pass
    return False

# Comprobación de conexión
if check_internet():
    # Una vez comprobado si hay conexión: si el programa se ejecuta por primera vez tendremos que descargar
    # los modelos de traduccion, pero si ya se ha ejecutado no sera necesario
    # por lo que realizaremos una comprobación con un archivo de texto
    filename = 'ejecutado.txt'

    if not os.path.isfile(filename):
        # La descarga se ejecutará solo si el archivo no existe
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == ingles and x.to_code == español, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
        
        with open(filename, 'w') as f:
            f.write('ejecutado')
    
    texto = "Hello World"
    # Detecta el idioma del texto
    idioma_origen = detect(texto)
    # Traduce el texto a español 
    traductor = LibreTranslateAPI('https://libretranslate.org/')
    texto_traducido = traductor.translate(texto, idioma_origen, español)
    print(f"Texto traducido con conexion: {texto_traducido}")
    # '¡Hola Mundo!'  
else:
    # Si no se cuenta con conexion se hace uso de argos para la traducción
    texto = "Hello World"
    # Detecta el idioma del texto
    idioma_origen = detect(texto)
    # Si el idioma detectado correspone a los modelos descargados, se realizara la traducción
    if  idioma_origen == ingles:
        translatedText = argostranslate.translate.translate(texto, ingles, español)
        print(f"Texto traducido sin conexion: {translatedText}")
        # '¡Hola Mundo!'
    else:
        print("No se cuenta con ese idioma para traduccion offline")  



