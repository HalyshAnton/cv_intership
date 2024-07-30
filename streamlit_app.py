import io
import streamlit as st
import ocr
from zipfile import ZipFile


def preprocess_file(pdf_file):
    """
    Processes a PDF file to extract its content using OCR and returns a
    BytesIO object containing the processed document in DOCX format.

    Args:
        pdf_file (UploadedFile): A file-like object representing
        the PDF to be processed.

    Returns:
        BytesIO: A BytesIO object containing the processed DOCX document.
    """
    doc = ocr.main(pdf_file)

    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    pdf_bytes.seek(0)

    return pdf_bytes


def create_zip(docx_files):
    """
    Creates a ZIP file from a dictionary of DOCX files.

    Args:
        docx_files (dict): A dictionary where keys are filenames
        and values are BytesIO objects containing DOCX files.

    Returns:
        BytesIO: A BytesIO object containing the created ZIP file.
    """
    zip_buffer = io.BytesIO()

    with ZipFile(zip_buffer, 'a') as zip_file:
        for name, docx_file in docx_files.items():
            zip_file.writestr(name, docx_file.getvalue())

    zip_buffer.seek(0)

    return zip_buffer


def main():
    """
    Streamlit app main function to convert uploaded PDF files to DOCX
    and download them as a ZIP archive.
    """

    st.title("PDF to DOCX Converter")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if st.button('Convert to docx'):
        with st.spinner("Processing..."):
            docx_files = {}
            for uploaded_file in uploaded_files:
                docx_file = preprocess_file(uploaded_file)

                new_name = uploaded_file.name.replace('.pdf', '.docx')
                docx_files[new_name] = docx_file

        zip_buffer = create_zip(docx_files)
        st.download_button("Download ZIP", zip_buffer,
                           file_name="converted_files.zip",
                           mime="application/zip")


if __name__ == "__main__":
    main()