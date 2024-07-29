import pymupdf
import numpy as np
import cv2
import os

def read_pdf(pdf_path):
    if not os.path.isfile(pdf_path):
        return []

    pdf_doc = pymupdf.open(pdf_path)
    imgs = []

    for page in pdf_doc:
        pix = page.get_pixmap()

        img_shape = (pix.h, pix.w, pix.n)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(img_shape)

        imgs.append(img)

    return imgs


if __name__ == '__main__':
    imgs = read_pdf(r'C:\Programs\cv_intership\data\printed 2 pages.pdf')

    for img in imgs:
        print(f'{img.shape=}')
        cv2.imshow('hjg', img[:, :, ::-1])
        cv2.waitKey(0)

    img = cv2.imread(r'C:\Programs\cv_intership\data\handwriting1.jpg')
    cv2.imshow('hjg', img[:, :, ::-1])
    cv2.waitKey(0)