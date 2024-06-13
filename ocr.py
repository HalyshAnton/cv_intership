import re
import warnings
import argparse

import layoutparser as lp
import pytesseract
from docx import Document
from tqdm import tqdm

from data_preprocessor import preprocess_image
from read_pdf import read_pdf


warnings.filterwarnings('ignore')

model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
                                 label_map={0: "Text", 1: "Title", 2: "List",
                                            3:"Table", 4:"Figure"})

def merge_blocks(layout, thresh_width=5):
    """
    Merge intersecting lp.TextBlock from layout

    Args:
        layout (lp.Layout): layout with y-coordinate
        sorted TextBlocks

        thresh_width (int): minimum intersecting width
        in pixels needed to merge 2 TextBlock

    Return:
        lp.Layout: layout with merged blocks
    """
    if len(layout) == 0:
        return layout

    blocks = []
    current_block = layout[0]

    for block in layout:
        inter = current_block.intersect(block)
        inter_width = inter.block.y_2 - inter.block.y_1

        #if block.is_in(prep_block, center=True):
        if inter_width > thresh_width:
            current_block = current_block.union(block)
        else:
            blocks.append(current_block)
            current_block = block

    blocks.append(current_block)
    return lp.Layout(blocks)


def get_text(block, img):
    """
    Get text from image using pytesseract

    Args:
        block (lp.TextBlock): image block returned by lp model

        img (np.ndarray): whole image from what the block will be cut

    Returns:
        str: text from image block
    """

    segment_image = (block
                     .pad(left=5, right=5, top=5, bottom=5)
                     .crop_image(img))

    text = pytesseract.image_to_string(segment_image, config='--psm 6')

    text = text.replace('\n', ' ')

    return text


def add_text(document, text, text_type):
    """
    Add text to docx file(document) depending on its type.
    Doesn't add figure and its annotation

    Args:
        document (docx.Document): document text will be added

        text (str): text to be added to document

        text_type (str): type of text from layoutparser model.
        Can be Text, Title or List.

    Returns:
        None
    """

    if re.match('Figure [0-9]+\..*', text):
        return

    if text_type == 'Text':
        document.add_paragraph(text)

    elif text_type == 'Title':
        document.add_heading(text)

    elif text_type == 'List':
        # check whether layoutparser combine list elements
        for element in re.split('\* |\+ ', text):
            if element.strip() == '':
                continue
            document.add_paragraph(element, style='List Bullet')


def pdf_to_word(pdf_path, save=False):
    """
    Convert pdf file to docx file.

    Args:
        pdf_path (str): path to pdf file
        save (bool): whether to save docx file, default False

    Returns:
        docx.Document: resulted docx file
    """

    doc = read_pdf(pdf_path)
    document = Document()

    main_progress_bar = tqdm(doc, desc='Document Processing')
    for page in main_progress_bar:
        prep_img = preprocess_image(page)

        layout = model.detect(prep_img)

        text_blocks = lp.Layout([b for b in layout if b.type != 'Figure'])
        text_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
        text_blocks = merge_blocks(text_blocks)

        for block in text_blocks:
            text = get_text(block, prep_img)
            add_text(document, text, block.type)

    if save:
        filename = pdf_path[:-4] + '.docx'
        document.save(filename)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image preprocessing')

    parser.add_argument('--path', type=str,
                        default='results/Scan_202465_4io-R77sgBI (1).pdf',
                        help='path to pdf file')

    parser.add_argument('--save', type=bool, default=True,
                        help='whether to save docx file')

    args = parser.parse_args()
    pdf_to_word(args.path, save=args.save)