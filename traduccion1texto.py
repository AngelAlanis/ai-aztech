#Aquí se importan las librerías langdetect y libretranslatepy 
#que se usaran para identificar el idioma del texto y lo traduzcan.
from langdetect import detect
from libretranslatepy import LibreTranslateAPI

#Aquí se inicializan la variable texto con el texto que se obtuvo de
#la libreria Pytesseract y el idioma al que se va a traducir, 
#en este caso español.
textovar="Hello, world"
idiomaTraducir="es"

#Hace la conexion con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

#Se detecta el idioma del texto con langdetect y lo guarda en la variable 
#idiomaDetectado para usarla en la traducción
idiomaDetectado=detect(textovar)

#Se crea finción para detectar el idioma y comparar si está en español 
#o no
def traducirTexto (textovar, deteccionIdioma):
    #Se usa un if para detectar si el idioma es español no haga nada
    if idiomaDetectado=='es':
       print(f"El texto ya se encuentra en español: {textovar}")
    else:
        #Para traducir se usa la variable traductor que se conecta con la api y se pone de   #parametros el texto que se va a traducir, el idioma detectado y el idioma al que 
        #se va a traducir
        traduccion=traductor.translate(textovar,deteccionIdioma,idiomaTraducir)    
        #Se imprime el texto ya traducido.
        print(f"Texto original: {textovar} idioma: {idiomaDetectado}")
        print(f"Texto traducido: {traduccion}")

#Main
traducirTexto(textovar,idiomaDetectado)