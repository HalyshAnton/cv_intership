import numpy as np
from skimage.transform import rotate
import cv2

from read_pdf import read_pdf


def deskew_image(img, delta=5, max_angle=50):
    """
    Rotates the image so that the text is placed horizontally.

    :param img: np.ndarray of shape (N, M)
        input image

    :param delta: scalar
        angle step

    :param max_angle: scalar
        maximum possible value of the angle

    :return: np.ndarray of shape (N, M)
        rotated image
    """

    # help function
    def get_score(img, angle):
        """
        Returns score describing the height and
        steepness of the peaks in the histogram of row sums
        """

        rotated_img = rotate(img, angle)
        hist = np.sum(rotated_img, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score

    # select angle with the highest score
    scores = []
    angles = np.arange(-max_angle, max_angle + delta, delta)

    for angle in angles:
        score = get_score(img, angle)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]

    # rotate image
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_DEFAULT)

    return rotated


def preprocess_image(img, crop_margin=20):
    """
    Preprocess image for OCR.

    :param img: np.ndarray of shape (N, M, 3)
        input image in RGB format

    :param crop_margin: int
         margin for removing edges

    :return: np.ndarray of shape (K, L, 3)
        preprocessed image in RGB format
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # cv2.adaptiveThreshold doesn't support BINARY_TRUNC
    # so this is custom realization
    mask = cv2.adaptiveThreshold(img, 1,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                21, 5).astype(bool)
    img[mask] = 255

    img = cv2.fastNlMeansDenoising(img, None, 3, 7, 21)

    img = deskew_image(img)

    # cropping: remove edges of page
    img = img[
          crop_margin: -crop_margin,
          crop_margin: -crop_margin
          ]

    shape = img.shape
    img = img[:, :, np.newaxis]
    img = np.broadcast_to(img, shape + (3,))

    return img


def preprocessing(pdf_path, show=False):
    """
    Preprocess pages of pdf file for OCR.

    Warning:
    Cropping is using without resizing

    :param pdf_path: str
        path to pdf file

    :param show: bool
        whether to show results

    :return: list[np.ndarray]
        list of preprocessed images of pages
    """
    imgs = read_pdf(pdf_path)

    prep_imgs = [preprocess_image(img) for img in imgs]

    if show:
        for img, prep_img in zip(imgs, prep_imgs):
            cv2.imshow('Original', img[:, :, ::-1])
            cv2.imshow('Preprocessed', prep_img)

            cv2.waitKey(0)

    return prep_imgs


if __name__ == '__main__':
    pdf_path = 'Scan_202465_4io-R77sgBI (1).pdf'
    preprocessing(pdf_path=pdf_path, show=True)