from PIL import Image
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'


# read image and show
img = Image.open('data/printed2.jpg')
img.show()

img = np.array(img)

# ocr
text = pytesseract.image_to_string(img)
print(text)