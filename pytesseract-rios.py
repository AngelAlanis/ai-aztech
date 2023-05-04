import pytesseract as tess
from PIL import Image
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
img=Image.open("Resources\\cerrado.png")
txt=tess.image_to_string(img)
print(txt)
img2=Image.open("Resources\\vehiculos.jpg")
txt2=tess.image_to_string(img2)
print(txt2)
img3=Image.open("Resources\\tarjetas.jpg")
txt3=tess.image_to_string(img3)
print(txt3)