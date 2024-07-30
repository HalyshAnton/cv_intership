import io
import pymupdf
import numpy as np
import cv2
import PyPDF2


def pypdf2_to_pymupdf(pdf_reader):
    """
    Converts a PyPDF2 PdfReader object to a PyMuPDF document.

    Args:
        pdf_reader (PyPDF2.PdfReader): A PyPDF2 PdfReader object
        representing the PDF to be converted.

    Returns:
        pymupdf.Document: A PyMuPDF Document object.
    """
    pdf_bytes = io.BytesIO()

    pdf_writer = PyPDF2.PdfWriter()

    for page in pdf_reader.pages:
        pdf_writer.add_page(page)

    pdf_writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    fitz_doc = pymupdf.open(stream=pdf_bytes.read(), filetype="pdf")

    return fitz_doc


def read_pdf(pdf_file):
    """
    Reads a PDF file and extracts each page as an image using PyMuPDF.

    Args:
        pdf_file (UploadedFile): A file-like object representing
        the PDF to be processed.

    Returns:
        list: A list of numpy arrays, each representing
        an image of a page in the PDF.
    """

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_doc = pypdf2_to_pymupdf(pdf_reader)
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