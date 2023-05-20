# Aztech | Equipo de Inteligencia Artificial

## Descripción

Parte de la Inteligencia Artificial para el proyecto de Aztech.

## Requisitos previos
- [Python: 3.10.0](https://www.python.org/downloads/release/python-3100/) instalado en el sistema.
- [Tesseract-OCR: 5.3.1.20230401](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe)  instalado en sistema, preferentemente en la ruta 'C:\\Program Files para que al final esté ubicado en 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe''.
- Las dependencias de Python especificadas en [requeriments.txt](https://github.com/AngelAlanis/ai-aztech/blob/app/requirements.txt), ver la sección de Instalación para más detalles.
- **IMPORTANTE:** Es necesario tener los archivos ai_aztech_app.py, aztech.names, y aztech.pt en el proyecto y en una misma carpeta o ruta.



## Instalación
1. Clona el repositorio del proyecto: git clone https://github.com/tu-usuario/nombre-del-repo.git
2. Ve al directorio del proyecto: cd nombre-del-repo
3. Instala las dependencias de Python: pip install -r requirements.txt
4. Instala las dependencias de JavaScript: npm install

## Configuración
Antes de ejecutar el programa, asegúrate de realizar la siguiente configuración:

1. Configura la ruta de Tesseract-OCR en el archivo main.py: pytesseract.pytesseract.tesseract_cmd = 'ruta-al-tesseract-ocr', si seguiste la guía no debería haber necesidad de cambiar la ruta.

## Uso
1. Ejecuta el servidor Flask para procesar las imágenes: python main.py
2. Accede a la aplicación web en tu navegador: http://localhost:5000
3. Otorga los permisos de cámara requeridos por la página web.
4. La cámara tomará fotos en intervalos regulares y las enviará al servidor.
5. El servidor procesará las imágenes y devolverá los resultados de detección de texto.
