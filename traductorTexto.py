#Aquí se importan las librerías langdetect y libretranslatepy 
#que se usaran para identificar el idioma del texto y lo traduzcan.
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
import libretranslatepy as lt
import libretranslatepy

#Aquí se inicializan las variables con el texto que se va a traducir y el 
#idioma al que se va a traducir, en este caso español.
textoPrueba1="Hello, world!"
textoPrueba2="Olá Mundo!"
textoPrueba3="Bonjour le monde!"
textoPrueba4="Estoy en español"
idiomaATraducir="es"

#Hace la conexion con la api por medio de internet.
traductor = LibreTranslateAPI('https://libretranslate.org/')

#Se detecta el idioma del texto con langdetect y lo guarda en la variable 
#idiomaDetectado para usarla en la traducción
idiomaDetectado1=detect(textoPrueba1)
idiomaDetectado2=detect(textoPrueba2)
idiomaDetectado3=detect(textoPrueba3)
idiomaDetectado4=detect(textoPrueba4)
print(idiomaDetectado4)
#Se usa un if para detectar si el idioma es español no haga nada
if(idiomaDetectado1=='es'):
    print(f"El texto ya se encuentra en español: {textoPrueba1}")
if(idiomaDetectado2=='es'):
    print(f"El texto ya se encuentra en español: {textoPrueba2}")
if (idiomaDetectado3=='es'):
    print(f"El texto ya se encuentra en español: {textoPrueba3}")
if (idiomaDetectado4=='es'):
    print(f"El texto ya se encuentra en español: {textoPrueba4}")


#Para traducir se usa la variable traductor que se conecta con la api y se pone de   #parametros el texto que se va a traducir, el idioma detectado y el idioma al que 
#se va a traducir
traduccion1=traductor.translate(textoPrueba1,idiomaDetectado1,idiomaATraducir)
traduccion2=traductor.translate(textoPrueba2,idiomaDetectado2,idiomaATraducir)
traduccion3=traductor.translate(textoPrueba3,idiomaDetectado3,idiomaATraducir)

#Se imprime el texto ya traducido.
print(f"Texto original: {textoPrueba1} idioma: {idiomaDetectado1}")
print(f"Texto traducido: {traduccion1}")
print(f"Texto original: {textoPrueba2} idioma: {idiomaDetectado2}")
print(f"Texto traducido: {traduccion2}")
print(f"Texto original: {textoPrueba3} idioma: {idiomaDetectado3}")
print(f"Texto traducido: {traduccion3}")
