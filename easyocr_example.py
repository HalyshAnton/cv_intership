from PIL import Image
import numpy as np
import easyocr


# read image and resize
img = Image.open('data/handwriting2.jpg')
img = img.resize((256, 256))

img.show()

img = np.array(img)

# ocr
reader = easyocr.Reader(['en'])
results = reader.readtext(img)

text = ' '.join(result[-2] for result in results)
print(text)